import torch.nn as nn
from torch import Tensor

from .TemporalModule import TemporalModule, StepMode

class BatchNormTT1d(TemporalModule):
    def __init__(self, input_features: int, time_steps: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True):
        super().__init__(StepMode.SINGLE_STEP, time_steps=time_steps)
        self.bntt = nn.ModuleList(
            [
                nn.BatchNorm1d(
                    input_features, eps=eps, momentum=momentum, affine=affine
                )
                for _ in range(time_steps)
            ])
        for bn in self.bntt:
            bn.bias = None

    def forward_single_step(self, x: Tensor, t: int):
        return self.bntt[t].forward(x)

class BatchNormTT2d(TemporalModule):
    def __init__(self, input_features: int, time_steps: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True):
        super().__init__(StepMode.SINGLE_STEP, time_steps=time_steps)
        self.bntt = nn.ModuleList(
            [
                nn.BatchNorm2d(
                    input_features, eps=eps, momentum=momentum, affine=affine,
                )
                for _ in range(time_steps)
            ]
        )

        # Disable bias/beta of Batch Norm
        for bn in self.bntt:
            bn.bias = None

    def forward_single_step(self, x: Tensor, t: int):
        return self.bntt[t].forward(x)
