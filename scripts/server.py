import os
import torch
import numpy as np
import io
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ksampler import KSampler
import boto3
import json
import random

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

aws_session = boto3.Session(
  aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
  aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)
s3_bucket = os.environ.get('AWS_S3_BUCKET')
s3 = aws_session.resource('s3')
sqs = boto3.resource('sqs', region_name='us-east-1')
queue = sqs.get_queue_by_name(QueueName='DadnetStack-SDJob558630A1-mDFvpxTGLuMQ')

seed_everything(42)
config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
device = torch.device("cuda")
model = model.to(device)
sampler_ddim = DDIMSampler(model)
sampler_plms = PLMSSampler(model)
sampler_kdpm2 = KSampler(model, 'dpm_2')
sampler_kdpm2_a = KSampler(model, 'dpm_2_ancestral')
sampler_keuler = KSampler(model, 'euler')
sampler_keuler_a = KSampler(model, 'euler_ancestral')
sampler_kheun = KSampler(model, 'heun')
sampler_klms = KSampler(model, 'lms')
n_samples = 1
precision_scope = autocast
ddim_eta = 0.0
f = 8
default_steps = 50
default_scale = 7.5
default_H = 512
default_W = 512
default_C = 4

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

def get_query_arg(post_data, arg, default_value):
  if arg in post_data:
    return post_data[arg]
  else:
    return default_value

for message in queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=20):
  try:
    job_req = json.loads(message.body)
    jid = job_req['id']
    print(f'Starting: {jid}')
    prompt = job_req['prompt']
    W = get_query_arg(job_req, 'W', default_W)
    H = get_query_arg(job_req, 'H', default_H)
    C = get_query_arg(job_req, 'C', default_C)
    scale = get_query_arg(job_req, 'scale', default_scale)
    steps = get_query_arg(job_req, 'steps', default_steps)
    sampler_type = get_query_arg(job_req, 'sampler', 'plms')
    seed = get_query_arg(job_req, 'seed', 0)
    if (seed == 0):
      seed = random.randrange(0, np.iinfo(np.uint32).max)
    seed_everything(seed)

    sampler = sampler_plms
    if sampler_type == 'ddim':
      sampler = sampler_ddim
    elif sampler_type == 'plms':
      sampler = sampler_plms
    elif sampler_type == 'k_dpm_2_a':
      sampler = sampler_kdpm2_a
    elif sampler_type == 'k_dpm_2':
      sampler = sampler_kdpm2
    elif sampler_type == 'k_euler_a':
      sampler = sampler_keuler_a
    elif sampler_type == 'k_euler':
      sampler = sampler_keuler
    elif sampler_type == 'k_heun':
      sampler = sampler_kheun
    elif sampler_type == 'k_lms':
      sampler = sampler_klms
      
    data = [n_samples * [prompt]]
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = model.get_learned_conditioning(n_samples * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [C, H // f, W // f]
                    samples_ddim, _ = sampler.sample(S=steps,
                                                      conditioning=c,
                                                      batch_size=n_samples,
                                                      shape=shape,
                                                      verbose=False,
                                                      unconditional_guidance_scale=scale,
                                                      unconditional_conditioning=uc,
                                                      eta=ddim_eta,
                                                      x_T=None)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    x_sample = x_checked_image_torch[0]
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    image = Image.fromarray(x_sample.astype(np.uint8))
                    img_file = io.BytesIO()
                    image.save(img_file, format='PNG')
                    s3_object = s3.Object(s3_bucket, f'sd/{jid}.png')
                    s3_object.put(Body=img_file.getvalue(), ContentType='image/png')
  finally:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(f'Completed: {jid}')