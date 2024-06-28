import torch
from torch import nn


class GhostNetwork(nn.Module):
    def __init__(self, num_features, l0=32, activation=nn.ReLU):
        super(GhostNetwork, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x


class GhostNetworkWithNormalization(nn.Module):
    def __init__(
        self, num_features, l0=32, activation=nn.ReLU, normalization=nn.BatchNorm1d
    ):
        super(GhostNetworkWithNormalization, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.norm = normalization(num_features, track_running_stats=True)
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.norm(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x


class NormalizationLayer(nn.Module):
    def __init__(
        self, matching: bool = True, device: torch.device = torch.device("cpu")
    ):
        super(NormalizationLayer, self).__init__()
        self.matching = matching
        if matching:
            self.shift = torch.tensor(
                [
                    12.615314483642578,
                    0.0,
                    -0.00042932009091600776,
                    10.0,
                    3.0,
                    2.0,
                    4.006599192507565e-05,
                    -0.4567539691925049,
                    -0.019999638199806213,
                    -19.99979019165039,
                    -149.963134765625,
                    0.004197417292743921,
                    -0.0004279722925275564,
                ],
                dtype=torch.float32,
                device=device,
            )
            self.scale = torch.tensor(
                [
                    155663427.38468552,
                    25787.57421875,
                    0.000783147057518363,
                    2,
                    20,
                    40,
                    1.9999514701485168,
                    1.02848482131958,
                    0.03999875485897064,
                    39.99972915649414,
                    299.958740234375,
                    1301.8299090280198,
                    0.0007813300180714577,
                ],
                dtype=torch.float32,
                device=device,
            )

    def forward(self, x):
        if self.matching:
            x[:,] = (x[:,] - self.shift) / self.scale
        return x


class GhostNetworkWithManualNormalization(nn.Module):
    def __init__(
        self,
        num_features,
        l0=32,
        matching=True,
        activation=nn.ReLU,
        device: torch.device = torch.device("cpu"),
    ):
        super(GhostNetworkWithManualNormalization, self).__init__()
        self.matching = matching
        self.quant = torch.quantization.QuantStub()
        self.normalization = NormalizationLayer(matching, device)
        self.layer0 = nn.Linear(num_features, l0)
        self.activation = activation()
        self.output = nn.Linear(l0, 1)
        self.sigmoid = nn.Sigmoid()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.normalization(x)
        x = self.layer0(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.sigmoid(x)
        x = self.dequant(x)
        return x
