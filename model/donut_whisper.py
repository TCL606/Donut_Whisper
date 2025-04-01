import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
# from model.modeling_whisper import WhisperModel
import copy
from transformers import VisionEncoderDecoderModel

class DonutWhisper(nn.Module):
    def __init__(self, whisper_model, image_model_path):
        super().__init__()

        donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)
        
        self.image_encoder = donut_model.encoder
        self.audio_encoder = whisper_model.model.encoder
        self.decoder = whisper_model.model.decoder
        # self.image_linear = nn.Linear(donut_model.config.encoder.hidden_size, whisper_model.dims.n_audio_state)
        # self.mix_linear = nn.Linear(whisper_model.dims.n_audio_state, whisper_model.dims.n_audio_state)

        self.image_linear = nn.Linear(donut_model.config.encoder.hidden_size, whisper_model.config.d_model)
        self.mix_linear = nn.Linear(whisper_model.config.d_model, whisper_model.config.d_model)

    def forward(self, audios, spectrograms, images, labels, texts, **kwargs):
        # image_feat = self.image_encoder(pixel_values=images, output_attentions=True, output_hidden_states=True, return_dict=True).last_hidden_state
        if isinstance(images, list):
            raise NotImplementedError
            # image_feat = []
            # for img in images:
            #     image_feat.append(self.image_encoder(pixel_values=img.unsqueeze(0)).last_hidden_state)
            # image_feat = F.gelu(self.image_linear(image_feat))
        else:
            image_feat = self.image_encoder(pixel_values=images).last_hidden_state
            image_feat = F.gelu(self.image_linear(image_feat))

        # audio_feat = self.audio_encoder(spectrograms)
        audio_feat = self.audio_encoder(spectrograms).last_hidden_state
        # encoder_output = audio_feat 

        encoder_output = torch.cat((audio_feat, image_feat), dim=1)
        encoder_output = F.gelu(self.mix_linear(encoder_output))

        if labels is not None:
            # output = self.decoder(x=labels, xa=encoder_output)

            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output).last_hidden_state
            logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

            return logits, encoder_output
        else:
            return None, encoder_output