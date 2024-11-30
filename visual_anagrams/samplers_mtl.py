from tqdm import tqdm

import torch
import torch.nn.functional as F

from diffusers.utils.torch_utils import randn_tensor
from attn_utils import *
from .mtl import noise_vector_balancing, noise_vector_rectification
from .sepen import anti_seg_loss

@torch.no_grad()
def sample_stage_1(model,
                   prompt_embeds,
                   negative_prompt_embeds, 
                   views,
                   num_inference_steps=100,
                   guidance_scale=7.0,
                   generator=None,
                   device='cuda',
                   prompts=None,
                   attn_control=None
                   ):

    # Params
    num_images_per_prompt = 1
    device = torch.device(device)
    height = model.unet.config.sample_size
    width = model.unet.config.sample_size
    batch_size = 1      # TODO: Support larger batch sizes, maybe
    num_prompts = prompt_embeds.shape[0]
    assert num_prompts == len(views), \
        "Number of prompts must match number of views!"

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Setup timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    # Make intermediate_images
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        model.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    concept_idxs = []
    for i in range(len(prompts)):
        # specify the concept in the prompt manually if it is not the last word
        concept = prompts[i].split(' ')[-1]
        concept_ids = model.tokenizer(concept, return_tensors='pt')['input_ids'][0, :-1]
        ln = len(concept_ids)
        prompt_ids = model.tokenizer(prompts[i], return_tensors='pt')['input_ids'][0]
        for j in range(len(prompt_ids) - ln):
            if (prompt_ids[j:j + ln] == concept_ids).all():
                concept_idxs.append([j, j + ln])
                break

    for i, t in enumerate(tqdm(timesteps)):
        # Apply views to noisy_image
        viewed_noisy_images = []
        for view_fn in views:
            viewed_noisy_images.append(view_fn.view(noisy_images[0]))
        viewed_noisy_images = torch.stack(viewed_noisy_images)

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([viewed_noisy_images] * 2)
        model_input = model.scheduler.scale_model_input(model_input, t)

        # Predict noise estimate
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Invert the unconditional (negative) estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_uncond = torch.stack(inverted_preds)

        # Invert the conditional estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_text, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_text = torch.stack(inverted_preds)

        # Split into noise estimate and variance estimates
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Reduce predicted noise and variances
        noise_pred = noise_pred.view(-1,num_prompts,3,64,64)
        predicted_variance = predicted_variance.view(-1,num_prompts,3,64,64)

        bs, tn, cn, he, wi = noise_pred.shape
        alpha = torch.ones((bs, tn, 1), device=noise_pred.device, dtype=noise_pred.dtype)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        co = 1

        alpha = noise_vector_balancing(noise_pred, predicted_variance, 1, i, viewed_noisy_images, prompts, t)
        co = noise_vector_rectification(noise_pred, predicted_variance, alpha)
        alpha_ = alpha * co

        noise_pred = noise_pred.view(bs, tn, -1) # (B, t, C*H*W)
        predicted_variance = predicted_variance.view(bs, tn, -1)

        noise_pred = noise_pred * alpha_
        noise_pred = noise_pred.sum(dim=1).view(bs, cn, he, wi)
        predicted_variance = predicted_variance * alpha
        predicted_variance = predicted_variance.sum(dim=1).view(bs, cn, he, wi)

        
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1).to(device)

        # compute the previous noisy sample x_t -> x_t-1
        noisy_images = model.scheduler.step(
            noise_pred, t, noisy_images, generator=generator, return_dict=False
        )[0]

        if attn_control is not None and i <= 25:
            noisy_images = noisy_images.detach()
            noisy_images.requires_grad = True
            torch.set_grad_enabled(True)
            optimizer = torch.optim.SGD([noisy_images], lr=5)
            optimizer.zero_grad()
            model.unet.zero_grad()
            # Apply views to noisy_image
            viewed_noisy_images = []
            for view_fn in views:
                viewed_noisy_images.append(view_fn.view(noisy_images[0]))
            viewed_noisy_images = torch.stack(viewed_noisy_images)

            # Duplicate inputs for CFG
            # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
            model_input = torch.cat([viewed_noisy_images] * 2)
            model_input = model.scheduler.scale_model_input(model_input, timesteps[i + 1])

            # Predict noise estimate
            noise_pred = model.unet(
                model_input,
                timesteps[i + 1],
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]

            amaps = aggregate_viewed_attention(attn_control, 32, ['down', 'up'], views, concept_idxs)
            loss = anti_seg_loss(amaps)
            loss.backward()
            optimizer.step()
            
            torch.set_grad_enabled(False)
            

    # Return denoised images
    return noisy_images


@torch.no_grad()
def sample_stage_2(model,
                   image,
                   prompt_embeds,
                   negative_prompt_embeds, 
                   views,
                   num_inference_steps=100,
                   guidance_scale=7.0,
                   noise_level=50,
                   generator=None,
                   prompts=None
                   ):

    # Params
    batch_size = 1
    num_prompts = prompt_embeds.shape[0]
    height = model.unet.config.sample_size
    width = model.unet.config.sample_size
    device = model.device
    num_images_per_prompt = 1

    # For CFG
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # Get timesteps
    model.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.scheduler.timesteps

    num_channels = model.unet.config.in_channels // 2
    noisy_images = model.prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        num_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )

    # Prepare upscaled image and noise level
    image = model.preprocess_image(image, num_images_per_prompt, device)
    upscaled = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)

    noise_level = torch.tensor([noise_level] * upscaled.shape[0], device=upscaled.device)
    noise = randn_tensor(upscaled.shape, generator=generator, device=upscaled.device, dtype=upscaled.dtype)
    upscaled = model.image_noising_scheduler.add_noise(upscaled, noise, timesteps=noise_level)

    # Condition on noise level, for each model input
    noise_level = torch.cat([noise_level] * num_prompts * 2)

    # Denoising Loop
    for i, t in enumerate(tqdm(timesteps)):
        # Cat noisy image with upscaled conditioning image
        model_input = torch.cat([noisy_images, upscaled], dim=1)

        # Apply views to noisy_image
        viewed_inputs = []
        for view_fn in views:
            viewed_inputs.append(view_fn.view(model_input[0]))
        viewed_inputs = torch.stack(viewed_inputs)

        # Duplicate inputs for CFG
        # Model input is: [ neg_0, neg_1, ..., pos_0, pos_1, ... ]
        model_input = torch.cat([viewed_inputs] * 2)
        model_input = model.scheduler.scale_model_input(model_input, t)

        # predict the noise residual
        noise_pred = model.unet(
            model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            class_labels=noise_level,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Extract uncond (neg) and cond noise estimates
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Invert the unconditional (negative) estimates
        # TODO: pretty sure you can combine these into one loop
        inverted_preds = []
        for pred, view in zip(noise_pred_uncond, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_uncond = torch.stack(inverted_preds)

        # Invert the conditional estimates
        inverted_preds = []
        for pred, view in zip(noise_pred_text, views):
            inverted_pred = view.inverse_view(pred)
            inverted_preds.append(inverted_pred)
        noise_pred_text = torch.stack(inverted_preds)

        # Split predicted noise and predicted variances
        noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1] // 2, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1] // 2, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Combine noise estimates (and variance estimates)
        noise_pred = noise_pred.view(-1,num_prompts,3,256,256)
        predicted_variance = predicted_variance.view(-1,num_prompts,3,256,256)

        bs, tn, cn, he, wi = noise_pred.shape

        alpha = noise_vector_balancing(noise_pred, predicted_variance, 2, i, viewed_inputs, prompts, t)
        co = noise_vector_rectification(noise_pred, predicted_variance, alpha)
        alpha_ = alpha * co

        noise_pred = noise_pred.view(bs, tn, -1) # (B, t, C*H*W)
        predicted_variance = predicted_variance.view(bs, tn, -1)

        noise_pred = noise_pred * alpha_
        noise_pred = noise_pred.sum(dim=1).view(bs, cn, he, wi)
        predicted_variance = predicted_variance * alpha
        predicted_variance = predicted_variance.sum(dim=1).view(bs, cn, he, wi)

        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1).to(device)

        # compute the previous noisy sample x_t -> x_t-1
        noisy_images = model.scheduler.step(
            noise_pred, t, noisy_images, generator=generator, return_dict=False
        )[0]

    # Return denoised images
    return noisy_images