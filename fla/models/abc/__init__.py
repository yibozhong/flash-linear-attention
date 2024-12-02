# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.abc.configuration_abc import ABCConfig
from fla.models.abc.modeling_abc import ABCForCausalLM, ABCModel
from fla.models.abc.modeling_abc import ABCForImageClassification

AutoConfig.register(ABCConfig.model_type, ABCConfig)
AutoModel.register(ABCConfig, ABCModel)
AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM)


__all__ = ['ABCConfig', 'ABCForCausalLM', 'ABCModel', 'ABCForImageClassification']
