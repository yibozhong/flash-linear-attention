# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gsa.configuration_gsa import GSAConfig
from fla.models.gsa.modeling_gsa import GSAForCausalLM, GSAModel, GSAForImageClassification

AutoConfig.register(GSAConfig.model_type, GSAConfig)
AutoModel.register(GSAConfig, GSAModel)
AutoModelForCausalLM.register(GSAConfig, GSAForCausalLM, GSAForImageClassification)


__all__ = ['GSAConfig', 'GSAForCausalLM', 'GSAModel', 'GSAForImageClassification']
