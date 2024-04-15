#!/usr/bin/env python

"""Tests for Izhikevich neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


def cfg(neuron_type="RS"):
    return snn.Izhikevich.cfgs[neuron_type][0]


@pytest.fixture(scope="module")
def izhikevich_instance():
    return snn.Izhikevich(*cfg())


@pytest.fixture(scope="module")
def izhikevich_instance_surrogate():
    return snn.Izhikevich(*cfg(), surrogate_disable=True)


@pytest.fixture(scope="module")
def izhikevich_reset_zero_instance():
    return snn.Izhikevich(*cfg(), reset_mechanism="zero")


@pytest.fixture(scope="module")
def izhikevich_reset_none_instance():
    return snn.Izhikevich(*cfg(), reset_mechanism="none")


@pytest.fixture(scope="module")
def izhikevich_hidden_instance():
    return snn.Izhikevich(*cfg(), init_hidden=True)


@pytest.fixture(scope="module")
def izhikevich_hidden_reset_zero_instance():
    return snn.Izhikevich(*cfg(),
                          init_hidden=True, reset_mechanism="zero"
                          )


@pytest.fixture(scope="module")
def izhikevich_hidden_reset_none_instance():
    return snn.Izhikevich(
        *cfg(), init_hidden=True, reset_mechanism="none"
    )


class TestIzhikevich:
    def test_izhikevich(self, izhikevich_instance, input_):
        v, u, syn_exc, syn_inh = izhikevich_instance.init_izhikevich()

        syn_rec = []
        mem_rec = []
        spk_rec = []
        u_rec = []

        for i in range(2):
            u_rec.append(u)
            spk, u, v, syn_exc, syn_inh = izhikevich_instance(input_[i], u,v,syn_exc, syn_inh)
            syn_rec.append(syn_inh+syn_exc)
            mem_rec.append(v)
            spk_rec.append(spk)

        assert u_rec[1] == u_rec[0] + izhikevich_instance.a * (izhikevich_instance.b*mem_rec[0]-u_rec[0])
        assert mem_rec[1] == mem_rec[0] + 0.04*mem_rec[0] * mem_rec[0] + 5*mem_rec[0] +140 - u_rec[0] + syn_rec[1]
        assert spk_rec[0] == spk_rec[1]

    @pytest.mark.skip("IZH only has one reset mechanism")
    def test_izhikevich_reset(
        self,
        izhikevich_instance,
        izhikevich_reset_zero_instance,
        izhikevich_reset_none_instance,
    ):
        lif1 = izhikevich_instance
        lif2 = izhikevich_reset_zero_instance
        lif3 = izhikevich_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_izhikevich_init_hidden(self, izhikevich_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = izhikevich_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    @pytest.mark.skip("IZH only has one reset mechanism")
    def test_izhikevich_init_hidden_reset_zero(
        self, izhikevich_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = izhikevich_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]


    def test_izhikevich_init_hidden_input_tensor(
        self, izhikevich_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = izhikevich_hidden_reset_none_instance(input_)
            spk_rec.append(spk)

        assert spk_rec[0].shape == (2,1)
        assert spk_rec[0][0] == spk_rec[1][0]
        assert spk_rec[1][0] == spk_rec[0][0]

    @pytest.mark.skip("IZH only has one reset mechanism")
    def test_izhikevich_init_hidden_reset_none(
        self, izhikevich_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = izhikevich_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_izhikevich_cases(self, izhikevich_hidden_instance, input_):
        with pytest.raises(TypeError):
            izhikevich_hidden_instance(input_, "123")

    def test_izhikevich_compile_fullgraph(self, izhikevich_instance_surrogate, input_):
        explanation = dynamo.explain(izhikevich_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0
