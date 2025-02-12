import numpy as np

import nethook
import os
from matplotlib import pyplot as plt

import model_loader
import pipeline_no_ed
from PIL import Image
from transformers import CLIPTokenizer
import torch
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import math
import torch.fft as fft
set_units = []


def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    print('x_freq.shape:', x_freq.shape)
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

def visualize_feature_map(feature_map, batch_index=0, num_columns=8, save_path=None):
    """
    Visualize feature maps across channels.

    Arguments:
    - feature_map: The input feature map as a numpy array of shape [b, c, h, w].
    - batch_index: The index of the batch to visualize (default is 0).
    - num_columns: Number of columns to use for the grid visualization.
    - save_path: Path to save the visualization as an image file (optional).

    Returns:
    - None (displays the plot or saves it as an image).
    """
    # Select the feature map for the specified batch index
    feature_map = feature_map[batch_index]  # Shape: [c, h, w]
    num_channels = feature_map.shape[0]

    # Calculate the number of rows needed
    num_rows = (num_channels + num_columns - 1) // num_columns

    # Create a grid for visualization
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 2))
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(feature_map[i], cmap='viridis')  # Visualize each channel
        ax.axis('off')
        ax.set_title(f"Channel {i + 1}", fontsize=8)

    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Feature map visualization saved to {save_path}")
    else:
        plt.show()

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_ssim(dcp1, dcp2):
    # if dcp1.ndim != 2 or dcp2.ndim != 2:
    #     raise ValueError("DCP maps must be 2D grayscale arrays.")

        # Calculate SSIM
    ssim, _ = compare_ssim(dcp1, dcp2, full=True)
    return ssim

def calculate_correlation_coefficient(dcp1, dcp2):
    """
    Calculate the correlation coefficient between two DCP maps.
    Arguments:
    - dcp1: First DCP map as a numpy array with values in [0, 1].
    - dcp2: Second DCP map as a numpy array with values in [0, 1].
    Returns:
    - correlation: Pearson correlation coefficient between the two DCP maps.
    """
    # Flatten the arrays to 1D
    dcp1_flat = dcp1.flatten()
    dcp2_flat = dcp2.flatten()

    # Calculate the correlation coefficient using numpy
    correlation = np.corrcoef(dcp1_flat, dcp2_flat)[0, 1]
    return correlation

def calculate_histogram(image, bins=256, range=(0, 1)):
    """
    Calculate the normalized histogram of an image.
    Arguments:
    - image: Input image as a numpy array.
    - bins: Number of bins for the histogram.
    - range: Range of the pixel values.
    Returns:
    - hist: Normalized histogram as a 1D numpy array.
    """
    hist, _ = np.histogram(image, bins=bins, range=range, density=True)
    return hist.astype(np.float32)


def compare_histograms(dcp1, dcp2, bins=256):
    """
    Compare two histograms using Bhattacharyya Distance and EMD.
    Arguments:
    - dcp1: First DCP map as a numpy array.
    - dcp2: Second DCP map as a numpy array.
    - bins: Number of bins for the histograms.
    Returns:
    - bhattacharyya_distance: Bhattacharyya distance between the histograms.
    - emd_distance: Earth Mover's Distance (EMD) between the histograms.
    """
    # Calculate histograms
    hist1 = calculate_histogram(dcp1, bins=bins)
    hist2 = calculate_histogram(dcp2, bins=bins)

    # Bhattacharyya Distance
    bhattacharyya_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    # Earth Mover's Distance (EMD)
    # Prepare histograms for EMD (signature format: bin centers + weights)
    bin_centers = np.linspace(0, 1, bins).astype(np.float32)
    signature1 = np.column_stack((bin_centers, hist1))
    signature2 = np.column_stack((bin_centers, hist2))
    emd = cv2.EMD(signature1, signature2, cv2.DIST_L2)

    return bhattacharyya_distance, emd

def cal_Dark_Channel(im, width=15):
    im_dark = np.min(im, axis=2)
    border = int((width - 1) / 2)
    im_dark_1 = cv2.copyMakeBorder(im_dark, border, border, border, border, cv2.BORDER_DEFAULT)
    res = np.zeros(np.shape(im_dark))
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i][j] = np.min(im_dark_1[i: i + width, j: j + width])

    return res

def patchify(x):
    """
    x: (N, H, W, C)
    imgs: (N, T(h*w), C)
    """
    imgs = x.flatten(2).transpose(1, 2)  # NCHW -> NLC

    return imgs

def unpatchify(x, c, p):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
    """
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs
def zero_out_tree_units(data, model):
    data[:, set_units, :, :] = 0.0
    return data

###### modify the values of feature maps #####
def turn_off_tree_units(data, model):
    # data[:, :, set_units] = -5.0
    x = unpatchify(data, 384, 1)
    # x[:, set_units, :, :] = x[:, set_units, :, :] * 0.8
    # x[:, set_units, :, :] = -5.0
    hidden_mean = x.mean(1).unsqueeze(1)
    B = hidden_mean.shape[0]
    hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
    hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
    hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(
        2).unsqueeze(3)
    print(hidden_mean.shape)
    x = x * ((1.8 - 1) * hidden_mean + 1)
    # x[:, set_units, :, :] = Fourier_filter(x[:, set_units, :, :], threshold=16, scale=0.8)
    x = patchify(x)
    return x



def conver2image(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def voted_units(input, unit_num):
    # arr = np.array(input)
    vote_units = []
    for i in range(0, unit_num):
        unit_count = 0.0
        for j in range(0, len(input)):
            arr = np.array(input[j])
            if i in arr[:, 0]:
                unit_count = unit_count + 1
        if unit_count > len(input) / 2.5:
            # print('unit', str(i), 'is useful')
            vote_units.append(i)
            # for j in range(0, len(input)):
            #     arr = np.array(input[j])
            #     for z in range(0, arr.shape[0]):
            #         if i == arr[z, 0]:
            #             select_units.append([arr[z, 0], arr[z, 1]])
    return vote_units

def generate_voted_dict(vote_units, input):
    vote_units_value = {}
    for unit in vote_units:
        count = 0
        value = 0
        for i in range(0, len(input)):
            one = input[i]
            for j in one:
                if unit == j[0]:
                    # print(j)
                    count += 1
                    value += j[1]
        vote_units_value[unit] = value / count

    return sorted(vote_units_value.items(), key=lambda x:x[1])
# units = [0, 1, 2, 3, 4, 5, 6]
# units = [28, 20] # conv4
# units = [38, 20, 16, 21, 35, 9, 34, 1, 0, 3, 2, 8] # conv3
# units = [20, 23, 18, 22] # conv2
# units = range(0, 48)


# t = t.to(device=device)

# model.retain_layer(check_layer)
# img = model(sr, hr, t)
# acts = model.retained_layer(check_layer)

# print(acts.shape)

# def update_tree_units(data, model):
#     mean, std = torch.std_mean(data[:, units, :, :], dim=[1, 2], keepdim=True)
#     mean_new, std_new = torch.std_mean(acts[:, units, :, :], dim=[1, 2], keepdim=True)
#     data[:, units, :, :] = std_new * (data[:, units, :, :] - mean) / std + mean_new
#     return data

# model.edit_layer('encoder_water.conv1', rule=turn_off_tree_units)
# img, hr_recover, _, _, _, _ = model(sr, hr, t)

# image2show = conver2image(img)
# gt = conver2image(sr)
# psnr = Metrics.calculate_psnr(image2show, gt)
# print(psnr)
#
# model(img)
# ranking = []
# for num in units:
#     set_units.append(num)
#     model.edit_layer(check_layer, rule=turn_off_tree_units)
#     img, _, _, _ = model(sr, hr, t)
#     x_0_recover = q_sample_recover(sr, t, predict_noise=img)
# # print(img.shape)
#     image2show = conver2image(x_0_recover)
#     gt = conver2image(sr)
#     psnr = Metrics.calculate_psnr(image2show, gt)
#     ranking.append([num, psnr])
#     set_units.clear()
# ranking.sort(key=lambda x: x[1])
# print (*ranking, sep="\n")

######## final output ##########


# acts = model.retained_layer(check_layer)
# acts2 = model.retained_layer('blocks2.0')
# print(acts.shape)
# print(acts.shape)
# print(acts2[0, 0, 0])

######## side output ##########
# condition_x = torch.mean(sr, dim=1, keepdim=True)
# model.retain_layer(check_layer)
# condition_x = torch.mean(hr, dim=1, keepdim=True)
#
# img = model(torch.cat([condition_x, sr], dim=1), hr, t)
# r = model.retained_layer(check_layer)
# r = r.data.cpu().numpy()

# img = model(sr, hr, t)
# img = model(torch.cat([condition_x, sr], dim=1), hr, t)
# x_0_recover = q_sample_recover(sr, t, predict_noise=img)
# at = extract((1.0 - betas).cumprod(), t, img.shape)
# x0_t = (sr - img * (1 - at).sqrt()) / at.sqrt()
# at_next = torch.ones_like(at)
# xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * img



'''
if __name__ == '__main__':
    # label_file = open()
    units_list_pos = []
    units_list_neg = []
    unit_num = 48 * 2 ** 1
    check_layer = 'encoder_water.block2_control'
    semantic_thres = 0.005
    for i, line in enumerate(open(image_root + '/label.txt')):
        if i == 50:
            break
        full_image_name, label_ = line.split(' ')
        # print(full_image_name.split('/')[-1])

        sr = Image.open(os.path.join(image_root, 'sr_16_128', full_image_name.split('/')[-1])).convert("RGB")
        style = Image.open(os.path.join(image_root, 'style_128', full_image_name.split('/')[-1])).convert("RGB")
        hr = Image.open(os.path.join(image_root, 'hr_128', full_image_name.split('/')[-1])).convert("RGB")
        sr = totensor(sr) * 2 - 1
        hr = totensor(hr) * 2 - 1
        sr = torch.unsqueeze(sr, 0)
        hr = torch.unsqueeze(hr, 0)

        style = totensor(style) * 2 - 1
        style = torch.unsqueeze(style, 0)

        # t = torch.full((1,), 100, dtype=torch.long)
        # sr = q_sample(sr, t)
        # hr = q_sample(hr, t)

        sr = sr.to(device=device)
        hr = hr.to(device=device)
        style = style.to(device=device)

        chose_units_set_pos = []
        chose_units_set_neg = []
        denoise_fn = model
        r = p_sample_loop2(sr, style, int(label_.strip()), continous=False)
        img = conver2image(r)
        base_semantic_value = compute_semantic_dis(Image.fromarray(img), 'red style')
        print('base value:', base_semantic_value)
        ####### check every unit with loop ###########
        for i in range(0, unit_num):
            print('testing unit num:', i)
            set_units = [i]
            model.edit_layer(check_layer, rule=turn_off_tree_units)
            # model.retain_layer(check_layer)
            # model.retain_layer('blocks2.0')
            # img = model(sr, hr, t)
            # denoise_fn = model
            r = p_sample_loop2(sr, style, int(label_.strip()), continous=False)
            img = conver2image(r)
            # cv2.imwrite('./a.jpg', cv2.cvtColor(conver2image(r), cv2.COLOR_RGB2BGR))
            dis = compute_semantic_dis(Image.fromarray(img), 'red style')
            print('dis=', dis - base_semantic_value, 'curr=', dis, 'base=', base_semantic_value)
            if (dis - base_semantic_value) > semantic_thres:
                chose_units_set_pos.append([i, float(dis - base_semantic_value)])
            # elif (dis - base_semantic_value) < -semantic_thres:
            #     chose_units_set_neg.append(i)
        # print(chose_units_set_pos)
        # chose_units_set_pos.sort(key=lambda t:t[1])
        units_list_pos.append(chose_units_set_pos)
        # units_list_neg.append(chose_units_set_neg)
    print('positive units:', units_list_pos)
    # print('negative units:', units_list_neg)
    units = voted_units(units_list_pos, unit_num)
    r = generate_voted_dict(units, units_list_pos)
    print(units)
    print(r)
    print(np.array(r)[:, 0])
    # block2 units = [10, 14, 33, 45, 48, 49, 56, 58, 62, 63, 64, 73, 74, 78, 79, 83, 84, 85, 90, 91, 93, 94, 95]
    # block1 units = [0, 6, 11, 15, 17, 24, 25, 44, 47]
    # [(11, 0.01865234375), (15, 0.019073486328125), (17, 0.0548583984375)]


'''
if __name__ == '__main__':
    DEVICE = "cuda:0"
    print(f"Using device: {DEVICE}")

    save_root = '../experiments/checkpoints_250111_173922'
    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE, in_channels=6, out_channels=3,
                                                               image_size=256)
    diffusion = models['diffusion']
    # print(diffusion)
    checkpoint = torch.load(os.path.join(save_root, 'sd_diffusion_999.pth'), map_location=DEVICE)
    diffusion.load_state_dict(checkpoint['model'], strict=True)
    models['diffusion'] = diffusion
    clip = models['clip']
    if not isinstance(diffusion, nethook.InstrumentedModel):
        diffusion = nethook.InstrumentedModel(diffusion)

        # prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    prompt = "remove haze from underwater image"
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = False
    cfg_scale = 8  # min: 1, max: 14

    ## IMAGE TO IMAGE

    # input_image = None
    # Comment to disable image to image
    save_root2 = '../experiments/checkpoints_250111_173922'
    image_name = '5040.jpg'
    image_path = '../../../data/UIE-dataset/UIEBD/test/image/' + image_name
    image_root = '../../../data/LSUI/test_input'
    image_root2 = '../../../data/LSUI/test_gt'
    paths = [path for path in os.listdir(image_root)]
    input_image = Image.open(os.path.join(image_root, image_name))
    result_image = Image.open(os.path.join(save_root2, 'results/LSUI (copy)', image_name))
    gt_image = Image.open(os.path.join(image_root2, image_name)).resize((256, 256))

    real_psnr = calculate_psnr(np.array(result_image), np.array(gt_image))
    print(f'no modified image psnr: {real_psnr}')
    save_image_root = save_root + '/results/LSUI'
    # Higher values means more noise will be added to the input image, so the result will further from the input image.
    # Lower values means less noise is added to the input image, so output will be closer to the input image.
    strength = 0.9

    ## SAMPLER
    sampler = "rf"
    num_inference_steps = 5
    seed = 42

    check_layer = 'blocks.2'
    # units = range(0, 1)

    # dark channel extraction
    dark_result_image = cal_Dark_Channel(np.array(result_image) / 255)
    # 0: use to find out all units that can affect DCP
    # 1: use to visualize the DCP against the no modified ones
    # 2: use to find out which unit can really improve PSNR and SSIM
    test_all_units = 1
    if test_all_units == 0:
        dis_dict = {}
        # ####### check every unit with loop ###########
        for i in range(0, 384):
            print('testing unit num:', i)
            set_units = [i]

            diffusion.edit_layer(check_layer, rule=turn_off_tree_units)

            output_image = pipeline_no_ed.generate(
                prompt=prompt,
                uncond_prompt=uncond_prompt,
                strength=strength,
                input_image=input_image,
                do_cfg=do_cfg,
                cfg_scale=cfg_scale,
                sampler_name=sampler,
                n_inference_steps=num_inference_steps,
                seed=seed,
                models=diffusion,
                clip=clip,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )

            dark_modifed_image = cal_Dark_Channel(output_image / 255)
            # bhatt_dist, emd_dist = compare_histograms(dark_modifed_image, dark_result_image)
            corr_coeff =  calculate_correlation_coefficient(dark_modifed_image, dark_result_image)
            ssim = calculate_ssim(dark_modifed_image, dark_result_image)
            print(f"Correlation Coefficient: {corr_coeff}")
            print(f"SSIM: {ssim}")
            dis_dict[str(i)] = [corr_coeff, ssim]

        final = sorted(dis_dict.items(), key=lambda x: x[1][0], reverse=True)
        print(final)
    elif test_all_units == 1:
        units = [378, 345, 339, 333, 275, 242, 194, 164, 144, 92]
        # set_units = [i]
        set_units = units
        check_layers = ['blocks.0']
        # for check_layer in check_layers:
        # diffusion.retain_layer(check_layer)
        diffusion.edit_layer(check_layer, rule=turn_off_tree_units)

        output_image = pipeline_no_ed.generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            strength=strength,
            input_image=input_image,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler,
            n_inference_steps=num_inference_steps,
            seed=seed,
            models=diffusion,
            clip=clip,
            device=DEVICE,
            idle_device="cpu",
            tokenizer=tokenizer,
        )
        # acts = diffusion.retained_layer(check_layer)
        # acts = unpatchify(acts, 384, 1)
        # x_freq = fft.fftn(acts, dim=(-2, -1))
        # x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        # magnitude = torch.abs(x_freq).data.cpu().numpy()
        # acts = acts.data.cpu().numpy()
        # print(acts.shape)
        # visualize_feature_map(acts, save_path='../data/' + image_name + '_' + check_layer + '.jpg')
        dark_modifed_image = cal_Dark_Channel(output_image / 255)
            # Image.fromarray(dark_result_image * 255).convert('L').save('../data/' + image_name + '_dcp.jpg')
        # corr_coeff = calculate_ssim(dark_result_image, dark_modifed_image)
        # print(f"SSIM: {corr_coeff}")
        # Correlation Coefficient: 0.9894616106237757 --- 0.5679176808407624
        # SSIM  0.589 --- 0.13410
        plt.subplot(2, 2, 1)
        plt.imshow(result_image)

        plt.subplot(2, 2, 2)
        plt.imshow(dark_result_image)

        plt.subplot(2, 2, 3)
        plt.imshow(output_image)

        plt.subplot(2, 2, 4)
        plt.imshow(dark_modifed_image)
        # plt.savefig('../data/' + image_name + '_compare.png')
        # plt.subplot(3, 2, 5)
        # plt.imshow(np.log(1 + magnitude[0, 378]))
        # plt.subplot(3, 2, 6)
        # plt.imshow(np.log(1 + magnitude[0, 174]))
        plt.show()
    elif test_all_units == 2:
        # units = [283, 168, 381, 337, 368, 55, 246, 184, 51, 166, 333, 287, 150, 36, 225, 164, 50, 37, 134, 133, 112, 95, 81, 121, 231, 116, 264, 151, 66, 140, 328, 179, 30, 41, 107]
        # set_units = [i]
        candicate_unit = []

        check_layers = ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', 'blocks.6', 'blocks.7']
        for check_layer2 in check_layers:

            for unit in range(0, 384):
                set_units = [unit]
                diffusion.edit_layer(check_layer2, rule=turn_off_tree_units)

                output_image = pipeline_no_ed.generate(
                    prompt=prompt,
                    uncond_prompt=uncond_prompt,
                    strength=strength,
                    input_image=input_image,
                    do_cfg=do_cfg,
                    cfg_scale=cfg_scale,
                    sampler_name=sampler,
                    n_inference_steps=num_inference_steps,
                    seed=seed,
                    models=diffusion,
                    clip=clip,
                    device=DEVICE,
                    idle_device="cpu",
                    tokenizer=tokenizer,
                )

                # dark_modifed_image = cal_Dark_Channel(output_image / 255)
                # ssim = calculate_ssim(np.array(result_image), output_image)
                psnr = calculate_psnr(np.array(gt_image), output_image)
                print(f"layer {check_layer2}, unit No.{unit}, PSNR: {psnr}")
                if psnr - real_psnr > 1.5:
                    candicate_unit.append((check_layer2, unit, psnr))

        print(f'candicate units:{candicate_unit}')
# value: 0.5
# candicate units:[('blocks.2', 64, 27.257662869255487), ('blocks.3', 35, 27.494036853374272), ('blocks.3', 90, 27.200159398657906), ('blocks.5', 365, 27.25354354070207)]
# value: 0.1
# candicate units:[('blocks.1', 192, 27.21481769408082), ('blocks.2', 38, 27.45967270203499), ('blocks.2', 65, 27.197547215977625), ('blocks.2', 208, 27.241147933933426), ('blocks.4', 144, 27.24182200096034), ('blocks.4', 192, 27.307120589545065), ('blocks.5', 35, 27.21854824479374), ('blocks.6', 26, 27.212039715268972), ('blocks.6', 222, 27.20464151815915), ('blocks.7', 35, 27.422212650313455), ('blocks.7', 173, 28.22948883874612), ('blocks.7', 320, 27.19602689913378)]
# value: 1.2
# candicate units:[('blocks.0', 14, 27.24857095567423), ('blocks.4', 362, 27.557417284508)]
# value: 0.8
# candicate units:[('blocks.1', 249, 27.16102689003991), ('blocks.2', 282, 27.249082995572294), ('blocks.4', 124, 27.16317947532531), ('blocks.6', 141, 27.238389938926126), ('blocks.7', 289, 27.165592593299305)]




