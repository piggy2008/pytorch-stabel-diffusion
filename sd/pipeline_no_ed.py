import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from rf import RectifiedFlow
from LRHR_dataset import LRHRDataset
from Scheduler import GradualWarmupScheduler
import torch.nn as nn
import os
import logging
from diffusion import Diffusion
from PIL import Image

# from accelerate import Accelerator
from accelerate.utils import set_seed
from Dark_channel_loss import DarkChannelLoss
from CDPRs import DistillationGuidedRouting
WIDTH = 256
HEIGHT = 256
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
        prompt,
        uncond_prompt=None,
        input_image=None,
        model_name='DiT',
        timestamps=2000,
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        n_inference_steps=50,
        models=None,
        clip=None,
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
):
    with (torch.no_grad()):
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
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        elif sampler_name == 'rf':
            sampler = RectifiedFlow()
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

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

        diffusion = models
        diffusion.to(device)
        latents = torch.randn_like(input_image_tensor, device=device)

        # timesteps = tqdm(torch.from_numpy(np.arange(0, n_inference_steps)[::-1].copy()))
        d_step = 1.0 / n_inference_steps

        for j in tqdm(range(n_inference_steps)):
            if model_name == 'DiT':
                time_embedding = torch.tensor([j * d_step * timestamps]).to(device)

                # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = torch.cat([input_image_tensor, latents], dim=1)

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
            latents = sampler.euler(latents, model_output, d_step)

        images = rescale(latents, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def generate_all(
        prompt,
        uncond_prompt=None,
        input_image_root=None,
        image_path='',
        save_root='',
        strength=0.8,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name="ddpm",
        model_name='DiT',
        n_inference_steps=50,
        timestamps=2000,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
):
    with (torch.no_grad()):
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
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator, num_training_steps=timestamps)
            sampler.set_inference_timesteps(n_inference_steps)
        elif sampler_name == 'rf':
            sampler = RectifiedFlow()
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        diffusion = models["diffusion"]
        diffusion.to(device)
        # encoder = models["encoder"]
        # encoder.to(device)
        # decoder = models["decoder"]
        # decoder.to(device)
        for image_name in image_path:
            image = Image.open(os.path.join(input_image_root, image_name))

            input_image_tensor = image.resize((WIDTH, HEIGHT))
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
            # encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            # condition = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            # sampler.set_strength(strength=strength)
            # latents = sampler.add_noise(latents, sampler.timesteps[0])

            latents = torch.randn(input_image_tensor.shape, generator=generator, device=device)
            if sampler_name == 'ddpm':
                timesteps = tqdm(sampler.timesteps)
                for i, timestep in enumerate(timesteps):
                    # (1, 320)
                    if model_name == 'DiT':
                        time_embedding = torch.tensor([timestep]).to(device)
                    else:
                        time_embedding = get_time_embedding(timestep, 'test').to(device)
                    # (Batch_Size, 4, Latents_Height, Latents_Width)
                    model_input = torch.cat([input_image_tensor, latents], dim=1)

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

            else:
                # timesteps = tqdm(torch.from_numpy(np.arange(0, n_inference_steps)[::-1].copy()))
                d_step = 1.0 / n_inference_steps

                for j in tqdm(range(n_inference_steps)):
                    if model_name == 'DiT':
                        time_embedding = torch.tensor([j * d_step * timestamps]).to(device)
                    else:
                        time_embedding = get_time_embedding(timestep, 'test').to(device)
                        # (Batch_Size, 4, Latents_Height, Latents_Width)
                    model_input = torch.cat([input_image_tensor, latents], dim=1)

                    if do_cfg:
                        model_input = model_input.repeat(2, 1, 1, 1)

                    # model_output is the predicted noise
                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                    model_output = diffusion(model_input, context, time_embedding)
                    # output_cond = diffusion(model_input, context, time_embedding)
                    # output_uncond = diffusion(model_input, context[1].unsqueeze(0), time_embedding)
                    if do_cfg:
                        output_cond, output_uncond = model_output.chunk(2)
                        model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

                    # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
                    latents = sampler.euler(latents, model_output, d_step)
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
            # images = decoder(latents)
            images = rescale(latents, (-1, 1), (0, 255), clamp=True)
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
            images = images.permute(0, 2, 3, 1)
            images = images.to("cpu", torch.uint8).numpy()
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            Image.fromarray(images[0]).save(os.path.join(save_root, image_name))

        # to_idle(encoder)
        to_idle(diffusion)
        # to_idle(decoder)


def train(sampler_name="ddpm",
          prompt='',
          uncond_prompt='',
          n_timestamp=1000,
          models={},
          model_name='DiT',
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
          save_path='',
          resume_path=''):
    set_seed(44)
    # accelerator = Accelerator(device_placement=False, mixed_precision='no')
    # accelerator.print(f'device {str(accelerator.device)} is used.')
    # device = accelerator.device
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    dataset = LRHRDataset(dataroot=dataroot, datatype='img', split='train', data_len=-1, image_size=image_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)
    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator, num_training_steps=n_timestamp)
        # sampler.set_inference_timesteps(n_timestamp)
    elif sampler_name == "rf":
        sampler = RectifiedFlow()
    else:
        raise ValueError("Unknown sampler value %s. ")

    diffusion = models["diffusion"]
    if resume_path is not None:
        load_part_of_model(diffusion, resume_path, True)
    diffusion.to(device)

    # encoder = models["encoder"]
    # encoder.to(device)
    # encoder.eval()
    # decoder = models["decoder"]
    # decoder.to(device)
    # decoder.eval()
    clip = models["clip"]
    clip.to(device)
    clip.eval()

    loss_func = nn.MSELoss(reduction='mean').to(device)

    logger = logging.getLogger('base')

    optimizer = torch.optim.AdamW(
        diffusion.parameters(), lr=lr, weight_decay=1e-4)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=1000, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2., warm_epoch=epochs // 10,
        after_scheduler=cosineScheduler)

    # diffusion, optimizer, warmUpScheduler, data_loader = accelerator.prepare(diffusion, optimizer, warmUpScheduler, data_loader)
    os.makedirs(save_path, exist_ok=True)
    loss_list = []
    # num = 0
    for e in range(epochs):

        with tqdm(data_loader, dynamic_ncols=True) as tqdmDataLoader:
            for batch, data in enumerate(tqdmDataLoader):
                data_high = data['high'].to(device)
                data_low = data['low'].to(device)
                [b, c, h, w] = data_high.shape
                # encoder -> 3, 512, 512 to 4, 512//8, 512//8
                # encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                # image_en = encoder(torch.cat([data_high, data_low], dim=0), encoder_noise)
                # image_en, condition = image_en.chunk(2, dim=0)

                uncond_tokens = tokenizer.batch_encode_plus(
                    [uncond_prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip(uncond_tokens)
                uncond_context = uncond_context.repeat(b, 1, 1)

                cond_tokens = tokenizer.batch_encode_plus(
                    [prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                cond_context = clip(cond_tokens)
                cond_context = cond_context.repeat(b, 1, 1)
                context = torch.cat([cond_context, uncond_context])
                # data_concate = torch.cat([data_color, snr_map], dim=1)
                optimizer.zero_grad()

                if sampler_name == 'rf':
                    t = torch.rand(b).to(device)
                    noisy_image, noise = sampler.create_flow(data_high, t)
                    if model_name == 'DiT':
                        timestamps = (t * n_timestamp).long()
                    else:
                        timestamps = get_time_embedding_rf(t, device)
                else:
                    t = torch.randint(0, n_timestamp, (b,)).long()
                    # timestamps = get_time_embedding(t).to(device)
                    noisy_image, noise = sampler.add_noise(data_high, t)
                    if model_name == 'DiT':
                        # timestamps = t * n_timestamp
                        timestamps = t.to(device)
                        # print(timestamps)
                    else:
                        timestamps = get_time_embedding(t).to(device)

                input_image = torch.cat([data_low, noisy_image], dim=1)
                input_image = torch.cat([input_image, input_image.clone()], dim=0)
                timestamps = torch.cat([timestamps, timestamps.clone()], dim=0)
                noise_pred = diffusion(input_image, context, timestamps)
                if sampler_name == 'rf':
                    data_high = torch.cat([data_high, data_high.clone()], dim=0)
                    noise = torch.cat([noise, noise.clone()], dim=0)
                    loss = loss_func(noise_pred, data_high - noise)
                else:
                    loss = loss_func(noise_pred, noise)
                loss = loss.mean()
                loss.backward()
                # accelerator.backward(loss)
                optimizer.step()
                loss_list.append(loss.item())
                if batch % batch_print_interval == 0:
                    # print(f'[Epoch {e}] [batch {batch}] loss: {loss.item()}')
                    logger.info('[Epoch {}] [batch {}] loss: {}'.format(e, batch, loss.item()))

        warmUpScheduler.step()

        if e % checkpoint_save_interval == 0 or e == epochs - 1:
            print(f'Saving model {e} to {save_path}...')
            logger.info('Saving model {} to {}...'.format(e, save_path))
            save_dict = dict(model=diffusion.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=e,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'sd_diffusion_{e}.pth'))
            # accelerator.save_model(diffusion, os.path.join(save_path, f'sd_diffusion_{e}.pth'))

def train_cdrps(sampler_name="ddpm",
          prompt='',
          uncond_prompt='',
          n_timestamp=1000,
          models={},
          model_name='DiT',
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
          save_path='',
          resume_path=''):
    set_seed(44)
    # accelerator = Accelerator(device_placement=False, mixed_precision='no')
    # accelerator.print(f'device {str(accelerator.device)} is used.')
    # device = accelerator.device
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    dataset = LRHRDataset(dataroot=dataroot, datatype='img', split='train', data_len=-1, image_size=image_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)
    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator, num_training_steps=n_timestamp)
        # sampler.set_inference_timesteps(n_timestamp)
    elif sampler_name == "rf":
        sampler = RectifiedFlow()
    else:
        raise ValueError("Unknown sampler value %s. ")

    diffusion = models["diffusion"]
    if resume_path is not None:
        load_part_of_model(diffusion, resume_path, True)
    diffusion.to(device)

    # encoder = models["encoder"]
    # encoder.to(device)
    # encoder.eval()
    # decoder = models["decoder"]
    # decoder.to(device)
    # decoder.eval()
    clip = models["clip"]
    clip.to(device)
    clip.eval()

    layers_to_gate = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'blocks.7']  # Specify layers to gate
    gamma = 0.05
    dgr = DistillationGuidedRouting(diffusion, layers_to_gate, device)
    diffusion.eval()

    loss_func = nn.MSELoss(reduction='mean').to(device)
    dcp_loss_fn = DarkChannelLoss(patch_size=15, lambda_smooth=1e-4, weight=0.1).to(device)
    logger = logging.getLogger('base')

    optimizer = torch.optim.AdamW(
        [gate.lambdas for gate in dgr.gates.values()], lr=lr, weight_decay=1e-4)
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=1000, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2., warm_epoch=epochs // 10,
        after_scheduler=cosineScheduler)

    # diffusion, optimizer, warmUpScheduler, data_loader = accelerator.prepare(diffusion, optimizer, warmUpScheduler, data_loader)
    os.makedirs(save_path, exist_ok=True)
    loss_list = []
    # num = 0
    for e in range(epochs):

        with tqdm(data_loader, dynamic_ncols=True) as tqdmDataLoader:
            for batch, data in enumerate(tqdmDataLoader):
                data_high = data['high'].to(device)
                data_low = data['low'].to(device)
                [b, c, h, w] = data_high.shape
                # encoder -> 3, 512, 512 to 4, 512//8, 512//8
                # encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                # image_en = encoder(torch.cat([data_high, data_low], dim=0), encoder_noise)
                # image_en, condition = image_en.chunk(2, dim=0)

                uncond_tokens = tokenizer.batch_encode_plus(
                    [uncond_prompt], padding="max_length", max_length=77
                ).input_ids
                # (Batch_Size, Seq_Len)
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
                # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                uncond_context = clip(uncond_tokens)
                uncond_context = uncond_context.repeat(b, 1, 1)

                # cond_tokens = tokenizer.batch_encode_plus(
                #     [prompt], padding="max_length", max_length=77
                # ).input_ids
                # # (Batch_Size, Seq_Len)
                # cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
                # # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
                # cond_context = clip(cond_tokens)
                # cond_context = cond_context.repeat(b, 1, 1)
                # context = torch.cat([cond_context, uncond_context])
                # data_concate = torch.cat([data_color, snr_map], dim=1)
                optimizer.zero_grad()

                if sampler_name == 'rf':
                    t = torch.rand(b).to(device)
                    noisy_image, noise = sampler.create_flow(data_high, t)
                    if model_name == 'DiT':
                        timestamps = (t * n_timestamp).long()
                    else:
                        timestamps = get_time_embedding_rf(t, device)
                else:
                    t = torch.randint(0, n_timestamp, (b,)).long()
                    # timestamps = get_time_embedding(t).to(device)
                    noisy_image, noise = sampler.add_noise(data_high, t)
                    if model_name == 'DiT':
                        # timestamps = t * n_timestamp
                        timestamps = t.to(device)
                        # print(timestamps)
                    else:
                        timestamps = get_time_embedding(t).to(device)

                input_image = torch.cat([data_low, noisy_image], dim=1)
                noise_pred = dgr.model(input_image, uncond_context, timestamps)
                # input_image = torch.cat([input_image, input_image.clone()], dim=0)
                # timestamps = torch.cat([timestamps, timestamps.clone()], dim=0)

                if sampler_name == 'rf':
                    # data_high = torch.cat([data_high, data_high.clone()], dim=0)
                    # noise = torch.cat([noise, noise.clone()], dim=0)
                    loss = loss_func(noise_pred, data_high - noise)
                    one_step_high = sampler.euler(noisy_image, noise_pred, timestamps)
                    one_step = rescale(one_step_high, (-1, 1), (0, 1), clamp=True)
                    data_high_normal = rescale(data_high, (-1, 1), (0, 1), clamp=True)
                    loss_dcp = dcp_loss_fn(one_step, data_high_normal)
                    loss += loss_dcp

                    loss += gamma * sum(torch.norm(gate.lambdas, p=1) for gate in dgr.gates.values())
                else:
                    loss = loss_func(noise_pred, noise)
                loss = loss.mean()
                loss.backward()
                # accelerator.backward(loss)
                optimizer.step()
                loss_list.append(loss.item())
                if batch % batch_print_interval == 0:
                    # print(f'[Epoch {e}] [batch {batch}] loss: {loss.item()}')
                    logger.info('[Epoch {}] [batch {}] loss: {}'.format(e, batch, loss.item()))

        warmUpScheduler.step()

        if e % checkpoint_save_interval == 0 or e == epochs - 1:
            print(f'Saving model {e} to {save_path}...')
            logger.info('Saving model {} to {}...'.format(e, save_path))
            save_dict = dict(model=diffusion.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=e,
                             loss_list=loss_list)
            torch.save(save_dict,
                       os.path.join(save_path, f'sd_diffusion_{e}.pth'))

            torch.save({layer: gates.cpu() for layer, gates in dgr.gates.items()}, os.path.join(save_path, f'control_gates_{e}.pth'))
            print("Optimized control gates saved.")
            # accelerator.save_model(diffusion, os.path.join(save_path, f'sd_diffusion_{e}.pth'))


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep, phase='train'):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    if phase == 'train':
        x = timestep[:, None] * freqs[None]
    else:
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def get_time_embedding_rf(timestep, device):
    # Shape: (160,)
    timestep = timestep * 1000
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    freqs = freqs.to(device)
    x = timestep[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def load_part_of_model(new_model, src_model_path, s):
    src_model = torch.load(src_model_path)['model']
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict, strict=s)
    return new_model