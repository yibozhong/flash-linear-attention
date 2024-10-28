# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageClassification

from fla.models.rwkv6.configuration_rwkv6 import RWKV6Config
from fla.models.rwkv6.modeling_rwkv6 import RWKV6ForCausalLM, RWKV6Model, RWKV6ForImageClassification

AutoConfig.register(RWKV6Config.model_type, RWKV6Config)
AutoModel.register(RWKV6Config, RWKV6Model)
AutoModelForCausalLM.register(RWKV6Config, RWKV6ForCausalLM)
AutoModelForImageClassification.register(RWKV6Config, RWKV6ForImageClassification)

__all__ = ['RWKV6Config', 'RWKV6ForCausalLM', 'RWKV6Model', 'RWKV6ForImageClassification']
