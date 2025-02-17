import torch
import torch.nn as nn
from DiT import DiT
import torch.fft as fft
def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape

    center_h, center_w = H // 2, W // 2
    radius = int(threshold * min(H, W) / 2)  # Define swapping region

    # Create a mask to select frequencies within the given radius
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    dist = torch.sqrt((X - center_w) ** 2 + (Y - center_h) ** 2)
    mask = (dist < radius).float()
    mask = mask.repeat(C, 1, 1).to(x.device) * scale.view(-1, 1, 1) + 1 - mask.repeat(C, 1, 1).to(x.device)
    # mask = torch.ones((B, C, H, W)).cuda()

    # crow, ccol = H // 2, W // 2
    # mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

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

class ControlGate(nn.Module):
    """ Scalar control gate for selected channels in a layer. """
    def __init__(self, num_channels, selected_channels=None):
        super(ControlGate, self).__init__()
        self.selected_channels = selected_channels if selected_channels is not None else list(range(num_channels))
        self.lambdas = nn.Parameter(torch.ones(len(self.selected_channels)) * 1.5)  # Initialize only selected gates

    def forward(self, x):
        x = unpatchify(x, 384, 1)
        gate_mask = torch.zeros(x.shape[1], device=x.device)
        gate_mask[self.selected_channels] = self.lambdas
        x = Fourier_filter(x, 0.3, gate_mask.view(-1, 1, 1))
        x = patchify(x)
        return x  # Apply gate only to selected channels


class DistillationGuidedRouting:
    def __init__(self, model, layers_to_gate, selected_channels, device):
        self.model = model
        self.layers_to_gate = layers_to_gate
        self.selected_channels = selected_channels
        # Replace layers with control gates
        self.gates = {}
        for idx in layers_to_gate:
            self.gates[idx] = ControlGate(getattr(self.model, 'blocks')[idx].mlp2.fc2.out_features, selected_channels).to(device)

        for idx, gate in self.gates.items():
            layer = getattr(self.model, 'blocks')[idx].mlp2.fc2
            # self.model['blocks'][idx].mlp2.fc2 = nn.Sequential(layer, gate)
            # setattr(self.model, layer, nn.Sequential(layer, gate))
            self.model.blocks[idx].mlp2.fc2 = nn.Sequential(layer, gate)

if __name__ == '__main__':
    layers_to_gate = range(0, 8) # Specify layers to gate
    gamma = 0.05
    # diffusion = DiT(depth=8, in_channels=6, out_channels=3, hidden_size=384, patch_size=4, num_heads=6, input_size=256)
    # # print(diffusion)
    # select_channels = [378, 345, 339, 333, 275, 242, 194, 164, 144, 92]
    # dgr = DistillationGuidedRouting(diffusion, layers_to_gate, select_channels, 'cuda:0')
    x = torch.zeros([2, 384, 64, 64])
    l = nn.Parameter(torch.ones(10))
    gate_mask = torch.zeros(x.shape[1], device=x.device)
    select_channels = [378, 345, 339, 333, 275, 242, 194, 164, 144, 92]
    gate_mask[select_channels] = l
    x = Fourier_filter(x, 0.3, gate_mask)