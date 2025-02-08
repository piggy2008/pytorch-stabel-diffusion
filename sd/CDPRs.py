import torch
import torch.nn as nn
from DiT import DiT

class ControlGate(nn.Module):
    """ Scalar control gate for each channel in a layer. """
    def __init__(self, num_channels=384):
        super(ControlGate, self).__init__()
        self.lambdas = nn.Parameter(torch.ones(num_channels))  # Initialize all gates to 1

    def forward(self, x):
        return x * self.lambdas.view(1, 1, -1)  # Apply gate channel-wise


class DistillationGuidedRouting:
    def __init__(self, model, layers_to_gate, device):
        self.model = model
        self.layers_to_gate = layers_to_gate

        # Replace layers with control gates
        self.gates = {}
        for idx in layers_to_gate:
            self.gates[idx] = ControlGate(getattr(self.model, 'blocks')[idx].mlp2.fc2.out_features).to(device)

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
    dgr = DistillationGuidedRouting(diffusion, layers_to_gate, 'cuda:0')
