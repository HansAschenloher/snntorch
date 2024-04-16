import torch.nn as nn
import torch

from .neurons import SpikingNeuron, LIF
from typing import Union


class Izhikevich(LIF):
    cfgs = {
        'RS': [[0.02, 0.2, -65, 8], [-70, -14]],
        'IB': [[0.02, 0.2, -55, 4], [-70, -14]],
        'CH': [[0.02, 0.2, -50, 2], [-70, -14]],
        'LTS': [[0.02, 0.25, -65, 2], [-64.4, -16.1]],
        'TC': [[0.02, 0.25, -65, 0.05], [-64.4, -16.1]],
        'FS': [[0.1, 0.2, -65, 2], [-70, -14]],
        'RZ': [[0.1, 0.25, -65, 2], [-64.4, -16.1]]
    }

    def __init__(self,
                 a: Union[float, torch.Tensor],
                 b: Union[float, torch.Tensor],
                 c: Union[float, torch.Tensor],
                 d: Union[float, torch.Tensor],
                 initial_u=-14.0,
                 initial_v=-70.0,
                 num_neurons=1,
                 learn_abcd=False,
                 time_resolution=1,
                 threshold=30.0,
                 use_psp=True,
                 alpha=0.9,
                 beta=0.8,
                 spike_grad=None,
                 surrogate_disable=False,
                 init_hidden=False,
                 inhibition=False,
                 learn_threshold=False,
                 reset_mechanism="zero",
                 state_quant=False,
                 output=False,
                 graded_spikes_factor=1.0,
                 learn_graded_spikes_factor=False,
                 log_spikes=False,
                 ):
        super().__init__(
            0.85,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            init_hidden=init_hidden,
            inhibition=inhibition,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
            log_spikes=log_spikes
        )

        self._register_buffer(a, b, c, d, learn_abcd, initial_u, initial_v, num_neurons)
        self.register_buffer("time_resolution", torch.Tensor([time_resolution]))
        self.use_psp = use_psp

        self.register_buffer("alpha", torch.as_tensor(alpha))
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("syn_exc", torch.as_tensor(0))
        self.register_buffer("syn_inh", torch.as_tensor(0))
        self.register_buffer("initial_u", torch.as_tensor(initial_u))
        self.register_buffer("initial_v", torch.as_tensor(initial_v))

    def init_izhikevich(self):
        self.reset_mem()
        return self.v, self.u, self.syn_exc, self.syn_inh

    def _register_buffer(self, a, b, c, d, learn_abcd, initial_u, initial_v, num_neurons):
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(float(a))
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(float(b))
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(float(c))
        if not isinstance(d, torch.Tensor):
            d = torch.as_tensor(float(d))
        if learn_abcd:
            self.a = nn.Parameter(a)
            self.b = nn.Parameter(b)
            self.c = nn.Parameter(c)
            self.d = nn.Parameter(d)
        else:
            self.register_buffer("a", a)
            self.register_buffer("b", b)
            self.register_buffer("c", c)
            self.register_buffer("d", d)
        self.register_buffer("u", torch.as_tensor([initial_u] * num_neurons))
        self.register_buffer("v", torch.as_tensor([initial_v] * num_neurons))

    def reset_mem(self):
        self.syn_exc = torch.zeros_like(
            self.syn_exc, device=self.syn_exc.device
        )
        self.syn_inh = torch.zeros_like(
            self.syn_inh, device=self.syn_inh.device
        )
        self.u = torch.ones_like(
            self.v, device=self.u.device
        ) * self.initial_u
        self.v = torch.ones_like(
            self.v, device=self.v.device
        ) * self.initial_v

    def forward(self, input: torch.Tensor,
                u: torch.Tensor = None,
                v: torch.Tensor = None,
                syn_exc: torch.Tensor = None,
                syn_inh: torch.Tensor = None):

        if (self.init_hidden and u != None):
            if not isinstance(u, torch.Tensor):
                u = torch.tensor(u)
            self.u = u
        if (self.init_hidden and v != None):
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.v = v
        if (self.init_hidden and syn_exc != None and self.use_psp):
            if not isinstance(syn_exc, torch.Tensor):
                syn_exc = torch.tensor(syn_exc)
            self.syn_exc = syn_exc
        if (self.init_hidden and syn_inh != None and self.use_psp):
            if not isinstance(syn_inh, torch.Tensor):
                syn_inh = torch.tensor(syn_inh)
            self.syn_inh = syn_inh

        spk = 0
        for i in range(int(self.time_resolution)):
            self.reset = self.mem_reset(self.v)
            self.u, self.v, self.syn_exc, self.syn_inh = self.update_hidden(input, self.u, self.v, self.syn_exc,
                                                                            self.syn_inh)
            if spk == []:
                spk = self.fire(self.v)
            else:
                spk += self.fire(self.v)

        if (self.init_hidden):
            if (self.output):
                return spk, self.v
            return spk
        else:
            return spk, self.u, self.v, self.syn_exc, self.syn_inh

    def update_hidden(self, input_, u, v, syn_exc, syn_inh):
        u, v, syn_exc, syn_inh = self.update_state(input_, u, v, syn_exc, syn_inh)
        u += self.reset * self.d
        v -= self.reset * (v - self.c)

        return u, v, syn_exc, syn_inh

    def update_state(self, input_, u, v, syn_exc, syn_inh):
        # return u, v, syn_exc, syn_inh
        if (self.use_psp):
            syn_exc = self.alpha * syn_exc + input_
            syn_inh = self.beta * syn_inh - input_
            dv = 0.04 * v * v + 5 * v + 140 - u + syn_exc + syn_inh
        else:
            dv = 0.04 * v * v + 5 * v + 140 - u + input_
        du = self.a * (self.b * v - u)
        return u + du, v + dv, syn_exc, syn_inh
        # return u + du / self.time_resolution, v + dv / self.time_resolution, syn_exc, syn_inh

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through
        time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Izhikevich):
                cls.instances[layer].syn_exc.detach_()
                cls.instances[layer].syn_inh.detach_()
                cls.instances[layer].u.detach_()
                cls.instances[layer].v.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Izhikevich):
                cls.instances[layer].syn_exc = torch.zeros_like(
                    cls.instances[layer].syn_exc,
                    device=cls.instances[layer].syn_exc.device,
                )
                cls.instances[layer].syn_inh = torch.zeros_like(
                    cls.instances[layer].syn_inh,
                    device=cls.instances[layer].syn_inh.device,
                )
                cls.instances[layer].u = torch.zeros_like(
                    cls.instances[layer].u,
                    device=cls.instances[layer].u.device,
                )
                cls.instances[layer].v = torch.zeros_like(
                    cls.instances[layer].v,
                    device=cls.instances[layer].v.device,
                )
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )

    def extra_repr(self) -> str:
        return f'a={self.a}, b={self.b}, c={self.c}, d={self.d}, use_syn={self.use_psp}, output={self.output}, hidden_mem={self.init_hidden}'
