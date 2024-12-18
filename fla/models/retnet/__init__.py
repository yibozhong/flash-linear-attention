# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.retnet.configuration_retnet import RetNetConfig
from fla.models.retnet.modeling_retnet import RetNetForCausalLM, RetNetModel, RetNetForImageClassification, RetNetForVideoClassification

AutoConfig.register(RetNetConfig.model_type, RetNetConfig)
AutoModel.register(RetNetConfig, RetNetModel)
AutoModelForCausalLM.register(RetNetConfig, RetNetForCausalLM, RetNetForImageClassification)


__all__ = ['RetNetConfig', 'RetNetForCausalLM', 'RetNetModel', 'RetNetForImageClassification', 'RetNetForVideoClassification']
