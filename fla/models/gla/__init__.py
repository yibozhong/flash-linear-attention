# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gla.configuration_gla import GLAConfig
from fla.models.gla.modeling_gla import GLAForCausalLM, GLAModel, GLAForImageClassification

AutoConfig.register(GLAConfig.model_type, GLAConfig)
AutoModel.register(GLAConfig, GLAModel)
AutoModelForCausalLM.register(GLAConfig, GLAForCausalLM, GLAForImageClassification)


__all__ = ['GLAConfig', 'GLAForCausalLM', 'GLAModel', 'GLAForImageClassification']
