import torch
import torch.nn as nn
from DiT import DiT

class ControlGate(nn.Module):
    """ Scalar control gate for selected channels in a layer. """
    def __init__(self, num_channels, selected_channels=None):
        super(ControlGate, self).__init__()
        self.selected_channels = selected_channels if selected_channels is not None else list(range(num_channels))
        self.lambdas = nn.Parameter(torch.ones(len(self.selected_channels)))  # Initialize only selected gates

    def forward(self, x):
        gate_mask = torch.zeros(x.shape[-1], device=x.device)
        gate_mask[self.selected_channels] = self.lambdas
        return x * gate_mask.view(1, 1, -1)  # Apply gate only to selected channels


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
    diffusion = DiT(depth=8, in_channels=6, out_channels=3, hidden_size=384, patch_size=4, num_heads=6, input_size=256)
    # print(diffusion)
    select_channels = [378, 345, 339, 333, 275, 242, 194, 164, 144, 92]
    dgr = DistillationGuidedRouting(diffusion, layers_to_gate, select_channels, 'cuda:0')
