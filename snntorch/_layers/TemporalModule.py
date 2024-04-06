import torch
from torch.nn import Module
from torch import Tensor
from enum import Enum, auto



__all__ = [
    "StepMode",
    "TemporalModule"
]

class StepMode(Enum):
    SINGLE_STEP = auto()
    MULTI_STEP = auto()


def convert_ANN_Modules_to_TemporalModules(module: Module):
    TemporalModule()

class TemporalModule(Module):
    # All Modules in the snn should be of this type
    # Has different modes for different modules
    # Has the methodes forward_single and forward_multi

    def __init__(self, step_mode: StepMode, time_steps: int):
        super().__init__()
        self._time_steps = time_steps
        self._step_mode: StepMode = step_mode

    @property
    def time_steps(self) -> int:
        return self._time_steps

    @time_steps.setter
    def time_steps(self, time_steps: int) -> None:
        self._time_steps = time_steps

    @property
    def step_mode(self):
        return self._step_mode

    @step_mode.setter
    def step_mode(self, step_mode: StepMode):
        self._step_mode: StepMode = step_mode

    def forward(self, x: Tensor):
        if self.step_mode == StepMode.SINGLE_STEP:
            ys = []
            for t in range(self.time_steps):
                y = self.forward_single_step(x[:,t], t)
                ys.append(y)

            #todo Stack tupels and number
            #TODO: rewrite Leaky Neuron
            #TODO: implement a clean eval print
            if(isinstance(ys[0], tuple)):
                output = []
                for i in range(len(ys[0])):
                    output.append(torch.stack(ys[i], dim=0))
                return output

            ys = torch.stack(ys, dim=1)
            return ys
        elif self.step_mode == StepMode.MULTI_STEP:
            return self.forward_multi_step(x)
    def forward_multi_step(self, x: Tensor):
        return x
    def forward_single_step(self, x: Tensor, t: int):
        return x
        m.backward = module.backward


    @staticmethod
    def from_ANN_module(module: Module, step_mode: StepMode ,time_steps: int):
        m = TemporalModule(time_steps=time_steps, step_mode=step_mode)
        m.forward_single_step = lambda x,t: module.forward(x)
        m.forward_multi_step = lambda x: module.forward(x)
        return m
