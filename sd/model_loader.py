from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from DiT import DiT

import model_converter

def preload_models_from_standard_weights(ckpt_path, device, in_channels=6, out_channels=3, selected_channels=[], image_size=512):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # diffusion = Diffusion().to(device)
    diffusion = DiT(depth=8, in_channels=in_channels, out_channels=out_channels,
                    hidden_size=384, patch_size=4, num_heads=6, input_size=image_size,
                    selected_channles=selected_channels)
    # diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }

if __name__ == '__main__':
    preload_models_from_standard_weights('../data/v1-5-pruned-emaonly.ckpt', 'cpu')