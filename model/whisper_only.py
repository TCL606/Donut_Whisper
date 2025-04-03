import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput


class WhisperOnlyConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class WhisperOnly(PreTrainedModel):
    def __init__(self, whisper_model):
        config = WhisperOnlyConfig()
        super().__init__(config)
        self.encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder

    def forward(self, input_ids=None, spectrograms=None, labels=None, **kwargs):
        encoder_output = self.encoder(spectrograms).last_hidden_state

        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output)
        else:
            decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output)
        
        output = decoder_output.last_hidden_state
        logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

        return ModelOutput(logits=logits, encoder_last_hidden_state=encoder_output)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "spectrograms": kwargs.get("spectrograms"),
        }


