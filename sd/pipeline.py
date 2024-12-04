import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from LRHR_dataset import LRHRDataset
from Scheduler import GradualWarmupScheduler
import torch.nn as nn
import os
import logging

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)
        x_t = torch.randn(latents_shape, generator=generator, device=device)
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = torch.cat([latents, x_t], dim=1)

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

def train(sampler_name="ddpm",
    uncond_prompt='',
    n_timestamp=1000,
    models={},
    seed=None,
    device=None,
    tokenizer=None,
    batch_size=10,
    epochs=100,
    lr=0.0001,
    batch_print_interval=100,
    checkpoint_save_interval=1,
    dataroot='',
    image_size=512,
    save_path='',):

    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    dataset = LRHRDataset(dataroot=dataroot, datatype='img', split='train', data_len=-1, image_size=image_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator, num_training_steps=n_timestamp)
        # sampler.set_inference_timesteps(n_timestamp)
    else:
        raise ValueError("Unknown sampler value %s. ")

    diffusion = models["diffusion"]
    diffusion.to(device)

    encoder = models["encoder"]
    encoder.to(device)
    decoder = models["decoder"]
    decoder.to(device)
    clip = models["clip"]
    clip.to(device)

    loss_func = nn.MSELoss(reduction='mean').to(device)

    logger = logging.getLogger('base')

    optimizer = torch.optim.AdamW(
        diffusion.parameters(), lr=lr, weight_decay=1e-4)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=1000, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2., warm_epoch=epochs // 10,
        after_scheduler=cosineScheduler)

    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
    os.makedirs(save_path, exist_ok=True)
    loss_list = []
    # num = 0
    for e in range(epochs):

        with tqdm(data_loader, dynamic_ncols=True) as tqdmDataLoader:
            for batch, data in enumerate(tqdmDataLoader):
                data_high = data['high'].to(device)
                data_low = data['low'].to(device)

                # data_concate = torch.cat([data_color, snr_map], dim=1)
                optimizer.zero_grad()
                # sampler.add_noise()
                [b, c, h, w] = data_high.shape
                t = torch.randint(0, n_timestamp, (b,)).long()

                # encoder -> 3, 512, 512 to 4, 512//8, 512//8
                encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                image_en = encoder(data_high, encoder_noise)
                noisy_image, noise = sampler.add_noise(image_en, t)
                timestamps = get_time_embedding(t).to(device)

                uncond_tokens = tokenizer.batch_encode_plus(
                    [uncond_prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip(uncond_tokens)
                uncond_context = uncond_context.repeat(b, 1, 1)
                noise_pred = diffusion(noisy_image, uncond_context, timestamps)
                loss = loss_func(noise_pred, noise)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                if batch % batch_print_interval == 0:
                    print(f'[Epoch {e}] [batch {batch}] loss: {loss.item()}')
                    logger.info('[Epoch {}] [batch {}] loss: {}'.format(e, batch, loss.item()))

        warmUpScheduler.step()

        if e % checkpoint_save_interval == 0 or e == e - 1:
            print(f'Saving model {e} to {save_path}...')
            logger.info('Saving model {} to {}...'.format(e, save_path))
            save_dict = dict(model=diffusion.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=e,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'sd_diffusion_{e}.pth'))

    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    if torch.is_tensor(timestep):
        x = timestep[:, None] * freqs[None]
    else:
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
