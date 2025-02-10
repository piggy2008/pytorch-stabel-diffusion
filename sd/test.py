import os.path

import model_loader
import pipeline
import pipeline_no_ed
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

DEVICE = "cuda:0"

# ALLOW_CUDA = True
# ALLOW_MPS = False
#
# if torch.cuda.is_available() and ALLOW_CUDA:
#     DEVICE = "cuda:3"
# elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
#     DEVICE = "mps"
print(f"Using device: {DEVICE}")



save_root = '../experiments/checkpoints_250111_173922'

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE, in_channels=6, out_channels=3, image_size=256)
diffusion = models['diffusion']

checkpoint = torch.load(os.path.join(save_root, 'sd_diffusion_999.pth'), map_location=DEVICE)

diffusion.load_state_dict(checkpoint['model'], strict=True)
models['diffusion'] = diffusion
## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "remove haze from underwater image"
uncond_prompt = ""  # Also known as negative prompt
do_cfg = False
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

# input_image = None
# Comment to disable image to image
image_name = '851.jpg'
image_path = '../../../data/UIE-dataset/UIEBD/test/image/' + image_name
image_root = '../../../data/LSUI/test_input'
paths = [path for path in os.listdir(image_root)]
input_image = Image.open(image_path)

save_image_root = save_root + '/results/LSUI'
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER

sampler = "rf"
num_inference_steps = 2
seed = 66666

# output_image = pipeline.generate(
#     prompt=prompt,
#     uncond_prompt=uncond_prompt,
#     input_image=input_image,
#     strength=strength,
#     do_cfg=do_cfg,
#     cfg_scale=cfg_scale,
#     sampler_name=sampler,
#     n_inference_steps=num_inference_steps,
#     seed=seed,
#     models=models,
#     device=DEVICE,
#     idle_device="cpu",
#     tokenizer=tokenizer,
# )

# Combine the input image and the output image into a single image.
# save_path = os.path.join(save_root, 'results')
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# Image.fromarray(output_image).save(os.path.join(save_path, image_name))

output_image = pipeline_no_ed.generate_all(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image_root=image_root,
    image_path=paths,
    save_root=save_image_root,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)