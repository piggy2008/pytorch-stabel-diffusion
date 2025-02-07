import torch
import torch.nn as nn
import torch.nn.functional as F


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


def compute_laplacian_matrix(image, epsilon=1e-6):
    """
    Compute the Laplacian matrix L for soft matting.

    Args:
    - image (Tensor): Input image tensor of shape (B, C, H, W).
    - epsilon (float): Small constant for numerical stability.

    Returns:
    - laplacian (Tensor): Laplacian-like matrix of shape (B, H, W).
    """
    B, C, H, W = image.shape
    mean_local = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)  # Local mean
    var_local = F.avg_pool2d(image ** 2, kernel_size=3, stride=1, padding=1) - mean_local ** 2  # Local variance

    laplacian = 1 / (var_local + epsilon)  # Inverse variance
    return laplacian


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

    # Define the loss function
    dcp_loss_fn = DarkChannelLoss(patch_size=15, lambda_smooth=1e-4, weight=0.1)

    # Compute the loss
    loss = dcp_loss_fn(predicted_image, input_image)
    print(f"Dark Channel Loss: {loss.item()}")