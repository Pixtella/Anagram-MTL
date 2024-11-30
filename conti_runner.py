import argparse
from pathlib import Path
from PIL import Image

import torch
from diffusers import DiffusionPipeline
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers_mtl import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata
from attn_utils import *
import os
from joblib import Parallel, delayed
import tqdm

stage_1 = None
stage_2 = None
stage_3 = None
generate_1024 = True
ctrl = None

def generate_anagram(style, prompts, views, save_dir, device='cuda', seed=0, num_inference_steps=30, guidance_scale=10.0, noise_level=50, view_args=None):
    global stage_1, stage_2, stage_3, generate_1024, ctrl
    # prepare the pipeline
    with torch.no_grad():
        if stage_1 is None:
            stage_1 = DiffusionPipeline.from_pretrained(
                    "DeepFloyd/IF-I-M-v1.0",
                    variant="fp16",
                    torch_dtype=torch.float16)
            stage_1.enable_model_cpu_offload()
            stage_1 = stage_1.to(device)
            ctrl = PerStepAttentionStore()
            register_attention_control(stage_1.unet, ctrl)

            stage_2 = DiffusionPipeline.from_pretrained(
                    "DeepFloyd/IF-II-M-v1.0",
                    text_encoder=None,
                    variant="fp16",
                    torch_dtype=torch.float16,
                )
            stage_2.enable_model_cpu_offload()
            stage_2 = stage_2.to(device)

            if generate_1024:
                stage_3 = DiffusionPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-x4-upscaler", 
                            torch_dtype=torch.float16
                        )
                stage_3.enable_model_cpu_offload()
                stage_3 = stage_3.to(device)

        views = get_views(views, view_args)
        prompts = [f'{style} {p}'.strip() for p in prompts]
        prompt_embeds = [stage_1.encode_prompt(p) for p in prompts]
        prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
        prompt_embeds = torch.cat(prompt_embeds)
        negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds
        ctrl.reset()

        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        sample_dir = save_dir / f'{seed:04}'
        sample_dir.mkdir(exist_ok=True, parents=True)

        generator = torch.manual_seed(seed)
        # Sample 64x64 image
        image = sample_stage_1(stage_1, 
                               prompt_embeds,
                               negative_prompt_embeds,
                               views,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale,
                               generator=generator,
                               device=device,
                               prompts=prompts,
                               attn_control=ctrl
                               )
                               
        save_illusion(image, views, sample_dir)

        # Sample 256x256 image, by upsampling 64x64 image
        image = sample_stage_2(stage_2,
                               image,
                               prompt_embeds,
                               negative_prompt_embeds, 
                               views,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale,
                               noise_level=noise_level,
                               generator=generator,
                               prompts=prompts
                               )
        save_illusion(image, views, sample_dir)

        if generate_1024:
            # Naively upsample to 1024x1024 using first prompt
            #   n.b. This is just the SD upsampler, and does not 
            #   take into account the other views. Results may be
            #   poor for the other view. See readme for more details
            image_1024 = stage_3(
                            prompt=prompts[0], 
                            image=image, 
                            noise_level=0,
                            output_type='pt',
                            generator=generator).images
            save_illusion(image_1024 * 2 - 1, views, sample_dir)