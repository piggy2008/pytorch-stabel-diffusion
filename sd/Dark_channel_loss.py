import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import torchvision.transforms as transforms
def dark_channel(image, patch_size=15):
    """
    Compute the dark channel of an image.

    Args:
    - image (Tensor): Input image tensor of shape (B, C, H, W), values in [0,1].
    - patch_size (int): Size of the local patch.

    Returns:
    - dark_channel (Tensor): Dark channel of shape (B, 1, H, W).
    """
    dark = torch.min(image, dim=1, keepdim=True)[0]  # Min across color channels (B, 1, H, W)
    dark_channel = F.max_pool2d(dark, kernel_size=patch_size, stride=1, padding=patch_size // 2)
    return dark_channel


def enhance_dark_channel(dark_channel_image, patch_size=15):
    """
    Enhance the dark channel by reducing haze effect using soft matting or max pooling.

    Args:
    - dark_channel_image (Tensor): The dark channel tensor, shape (B, 1, H, W).
    - patch_size (int): Size of the local patch used for enhancement.

    Returns:
    - enhanced_dark_channel (Tensor): Enhanced dark channel.
    """
    # Enhance the dark channel by using a soft matting-like technique (or pooling)
    # Applying a max pooling operation to reduce haze in dark regions
    enhanced_dark_channel = F.max_pool2d(dark_channel_image, kernel_size=patch_size, stride=1, padding=patch_size // 2)
    return enhanced_dark_channel

def compute_laplacian_matrix(image, epsilon=1e-6):
    """
    Compute the Laplacian matrix L for soft matting.

    Args:
    - image (Tensor): Input image tensor of shape (B, C, H, W).
    - epsilon (float): Small constant for numerical stability.

    Returns:
    - laplacian (Tensor): Laplacian-like matrix of shape (B, H, W).
    """

    mean_local = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)  # Local mean
    var_local = F.avg_pool2d(image ** 2, kernel_size=3, stride=1, padding=1) - mean_local ** 2  # Local variance

    laplacian = 1 / (var_local + epsilon)  # Inverse variance
    return laplacian


def soft_matting_refinement(dark_channel, image, lambda_matting=0.01):
    """
    Apply soft matting refinement to improve the dark channel.

    Args:
    - dark_channel (Tensor): Dark channel tensor of shape (B, 1, H, W).
    - image (Tensor): Original input image (B, 3, H, W).
    - lambda_matting (float): Strength of matting refinement.

    Returns:
    - refined_dark_channel (Tensor): Softly refined dark channel.
    """
    laplacian = compute_laplacian_matrix(image)  # Compute Laplacian weights
    refined_dark_channel = dark_channel - lambda_matting * (laplacian * dark_channel)
    return torch.clamp(refined_dark_channel, 0.0, 1.0)


def estimate_depth_from_dark_channel(dark_channel, gamma=0.8):
    """
    Estimate a depth-aware weighting map based on the dark channel intensity.

    Args:
    - dark_channel (Tensor): Dark channel tensor of shape (B, 1, H, W).
    - gamma (float): Strength of depth scaling.

    Returns:
    - depth_map (Tensor): Estimated depth weighting map.
    """
    depth_map = 1 - dark_channel  # Invert dark channel to get depth
    depth_map = depth_map ** gamma  # Apply non-linear scaling
    return torch.clamp(depth_map, 0.0, 1.0)


def depth_aware_dark_channel(dark_channel, image, gamma=0.8):
    """
    Enhance the dark channel using depth-awareness.

    Args:
    - dark_channel (Tensor): Dark channel tensor (B, 1, H, W).
    - image (Tensor): Original input image (B, 3, H, W).
    - gamma (float): Strength of depth scaling.

    Returns:
    - enhanced_dark_channel (Tensor): Depth-aware enhanced dark channel.
    """
    depth_map = estimate_depth_from_dark_channel(dark_channel, gamma)
    enhanced_dark_channel = dark_channel * depth_map  # Weight dark channel by depth
    return torch.clamp(enhanced_dark_channel, 0.0, 1.0)


class DarkChannelLoss(nn.Module):
    """
    Dark Channel Prior Loss with local smoothness and soft matting.
    """

    def __init__(self, patch_size=15, lambda_smooth=1e-4, weight=1.0):
        super(DarkChannelLoss, self).__init__()
        self.patch_size = patch_size
        self.lambda_smooth = lambda_smooth
        self.weight = weight

    def forward(self, predicted_image, input_image):
        """
        Compute the dark channel loss.

        Args:
        - predicted_image (Tensor): Output image from the network (B, 3, H, W), values in [0,1].
        - input_image (Tensor): Original hazy image (B, 3, H, W).

        Returns:
        - loss (Tensor): Scalar loss value.
        """
        # Compute dark channel of the predicted image
        dark_channel_pred = dark_channel(predicted_image, self.patch_size)

        # Compute dark channel of the input hazy image
        dark_channel_input = dark_channel(input_image, self.patch_size)
        dark_channel_input = soft_matting_refinement(dark_channel_input, input_image, lambda_matting=0.00005)
        # Fidelity term (ensure dark channel of output is small)
        fidelity_loss = torch.mean(dark_channel_pred)

        # Soft matting regularization (local smoothness)
        laplacian = compute_laplacian_matrix(input_image)
        smoothness_loss = torch.mean(laplacian * (dark_channel_pred - dark_channel_input) ** 2)

        # Total loss
        total_loss = self.weight * (fidelity_loss + self.lambda_smooth * smoothness_loss)
        return total_loss

# Example Usage
if __name__ == "__main__":
    # Example tensors (batch size 4, RGB channels, 256x256 image)
    predicted_image = torch.rand((4, 3, 256, 256))  # Simulated dehazed output
    input_image = torch.rand((4, 3, 256, 256))  # Simulated hazy input
    image_root2 = '../../../data/LSUI/test_gt'
    image_name = '5040.jpg'
    gt_image = Image.open(os.path.join(image_root2, image_name)).resize((256, 256))
    # Define the loss function
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(gt_image)
    img_tensor = img_tensor.unsqueeze(0)
    dark = dark_channel(img_tensor)
    # dark2 = enhance_dark_channel(dark)
    dark2 = soft_matting_refinement(dark, img_tensor, lambda_matting=0.00005)
    print(dark.shape)
    from matplotlib import pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(dark.data.cpu().numpy()[0, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(dark2.data.cpu().numpy()[0, 0])
    plt.show()
    # dcp_loss_fn = DarkChannelLoss(patch_size=15, lambda_smooth=1e-4, weight=0.1)

    # Compute the loss
    # loss = dcp_loss_fn(predicted_image, input_image)
    # print(f"Dark Channel Loss: {loss.item()}")