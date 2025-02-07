import torch
import torch.nn as nn

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
        self.gates = {layer: ControlGate(getattr(model, layer).out_channels).to(device)
                      for layer in layers_to_gate}
        for layer, gate in self.gates.items():
            setattr(self.model, layer, nn.Sequential(getattr(self.model, layer), gate))
