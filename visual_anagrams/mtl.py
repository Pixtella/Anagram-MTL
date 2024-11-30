import torch
import numpy as np
from typing import Tuple
from cvxopt import matrix, solvers
import tqdm

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
CLIP_model = None
processor = None

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

clip_model_noise_aware = None

def calc_alignment_noise_aware(imgs, texts, stage, step, ts):
    import torchvision.transforms as transforms
    resize_transform = transforms.Resize((64, 64))
    if imgs[0].shape[1] != 64:
        imgs = [resize_transform(img) for img in imgs]
        imgs = torch.stack(imgs)
    global clip_model_noise_aware
    device = 'cuda'
    if clip_model_noise_aware is None:
        clip_model_noise_aware = create_clip_model(device=device)
        clip_model_noise_aware.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
        clip_model_noise_aware.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))
    n = len(imgs)
    ret = np.zeros((n, n))
    t = torch.tensor([ts] * n, device=device)
    x = clip_model_noise_aware.image_embeddings(imgs, t).to(device)
    y = clip_model_noise_aware.text_embeddings(texts).to(device)
    ret = (x / x.norm(dim=-1, keepdim=True) @ (y / y.norm(dim=-1, keepdim=True)).T).cpu().numpy()
    return ret


def noise_vector_balancing(noise_pred: torch.Tensor, predicted_variance: torch.Tensor, stage, step, viewed_noisy_images: torch.Tensor, prompts, ts):

    b, t, c, h, w = noise_pred.shape
    noise_pred = noise_pred.view(b, t, -1) # (B, t, C*H*W)
    predicted_variance = predicted_variance.view(b, t, -1)

    if stage == 1:
        sim_mat = calc_alignment_noise_aware(viewed_noisy_images / 2. + 0.5, prompts, stage, step, ts)
    if stage == 2:
        sim_mat = calc_alignment_noise_aware(viewed_noisy_images[:, :3] / 2. + 0.5, prompts, stage, step, ts)
    
    diag = np.diag(sim_mat)
    alpha = np.power(diag, -1 - step / 30)
    alpha = torch.tensor(alpha).to(noise_pred.device).to(noise_pred.dtype).view(b, t, 1)
    alpha = alpha / alpha.sum(dim=1, keepdim=True)

    return alpha

def noise_vector_rectification(noise_pred: torch.Tensor, predicted_variance: torch.Tensor, alpha) -> Tuple[torch.Tensor, torch.Tensor]:
    # noise_pred, predicted_variance: (B, t, C, H, W)
    b, t, c, h, w = noise_pred.shape
    noise_pred = noise_pred.view(b, t, -1) # (B, t, C*H*W)
    predicted_variance = predicted_variance.view(b, t, -1)
    de = 0
    for i in range(t):
        de += alpha[0, i, 0] ** 2
        for j in range(i + 1, t):
            rho_ij = (noise_pred[0, i] * noise_pred[0, j]).mean().cpu().item()
            de += 2 * rho_ij * alpha[0, i, 0] * alpha[0, j, 0]
    c = 1 / (de ** 0.5)
    return c