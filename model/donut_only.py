import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import VisionEncoderDecoderModel
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

class DonutOnlyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class DonutOnly(PreTrainedModel):
    def __init__(self, image_model_path):
        config = DonutOnlyConfig()
        super().__init__(config)

        self.donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)

    def forward(self, input_ids=None, images=None, labels=None, **kwargs):
        
        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 2
        else:
            new_labels = input_ids

        outputs = self.donut_model(pixel_values=images, decoder_input_ids=new_labels)

        return outputs

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "images": kwargs.get("images"),
        }