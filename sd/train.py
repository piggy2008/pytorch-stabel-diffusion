import model_loader
import pipeline
# import pipeline_no_ed
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import yaml
import logging
import os
from datetime import datetime

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, '{}.log'.format(phase))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

config = yaml.load(open('./config/train.yaml', 'rb'), Loader=yaml.FullLoader)
DEVICE = config.get('device')
# ALLOW_CUDA = False
# ALLOW_MPS = False
# setting logger
save_path = config.get('save_path', '../experiments')
experiments_root = os.path.join(save_path, '{}_{}'.format('checkpoints', datetime.now().strftime('%y%m%d_%H%M%S')))
if not os.path.exists(experiments_root):
    os.makedirs(experiments_root)

setup_logger(None, experiments_root,
              'train', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
tokenizer = CLIPTokenizer(config.get('vocab_file'), merges_file=config.get('merge_file'))
model_file = config.get('pre_trained_param')
image_root = config.get('img_root_path')
image_size = config.get('image_size')
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE, image_size=(image_size // 4))
epochs = config.get('epochs')
batch_size = config.get('batch_size')
lr_adjust_epoch = config.get('lr_adjust_epoch')
lr = config.get('lr')
batch_print_interval = config.get('batch_print_interval')
checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
sampler = config.get('sampler_name')
seed = 42

logger.info('sampling method:' + sampler)
logger.info('training set root:' + image_root)
logger.info('pretrained file:' + model_file)
logger.info('training total epoch:' + str(epochs))
logger.info('batch size:' + str(batch_size))
logger.info('input image size:' + str(image_size))
logger.info('start learning rate:' + str(lr))

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = config.get('prompt')
uncond_prompt =  config.get('uncond_prompt')  # Also known as negative prompt
# do_cfg = True
# cfg_scale = 8  # min: 1, max: 14
logger.info('clip input prompt text(positive):' + prompt)
logger.info('clip input prompt text(negative):' + uncond_prompt)
## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
image_path = "../images/dog.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.9

## SAMPLER



output_image = pipeline.train(
    uncond_prompt=uncond_prompt,
    sampler_name=sampler,
    seed=seed,
    models=models,
    device=DEVICE,
    tokenizer=tokenizer,
    batch_size=batch_size,
    epochs=epochs,
    lr=lr,
    image_size=image_size,
    batch_print_interval=batch_print_interval,
    checkpoint_save_interval=checkpoint_save_interval,
    dataroot=image_root,
    save_path=experiments_root
)

# Combine the input image and the output image into a single image.
# Image.fromarray(output_image)