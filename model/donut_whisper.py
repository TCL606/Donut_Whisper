import os
import torch
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
# from model.modeling_whisper import WhisperModel
import copy
from transformers import VisionEncoderDecoderModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

class DonutWhisperConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()

class DonutWhisper(PreTrainedModel):
    def __init__(self, whisper_model, image_model_path):
        config = DonutWhisperConfig()
        super().__init__(config)

        donut_model = VisionEncoderDecoderModel.from_pretrained(image_model_path)

        self.image_encoder = donut_model.encoder
        self.audio_encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder

        self.image_linear = nn.Linear(donut_model.config.encoder.hidden_size, whisper_model.config.d_model)
        self.mix_linear = nn.Linear(whisper_model.config.d_model, whisper_model.config.d_model)

    def forward(self, input_ids=None, spectrograms=None, images=None, labels=None, past_key_values=None, use_cache=None, **kwargs):

        if isinstance(images, list):
            raise NotImplementedError
        else:
            image_feat = self.image_encoder(pixel_values=images).last_hidden_state
            image_feat = F.gelu(self.image_linear(image_feat))

        audio_feat = self.audio_encoder(spectrograms).last_hidden_state

        encoder_output = torch.cat((audio_feat, image_feat), dim=1)
        encoder_output = F.gelu(self.mix_linear(encoder_output))

        if labels is not None:
            new_labels = copy.deepcopy(labels)
            new_labels[labels == -100] = 50257
            decoder_output = self.decoder(input_ids=new_labels, encoder_hidden_states=encoder_output, past_key_values=past_key_values, use_cache=use_cache)
        else:
            decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_output, past_key_values=past_key_values, use_cache=use_cache)
        
        output = decoder_output.last_hidden_state
        logits = torch.matmul(output, self.decoder.embed_tokens.weight.t())

        return ModelOutput(logits=logits, encoder_last_hidden_state=encoder_output, past_key_values=decoder_output.past_key_values)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {
            "input_ids": input_ids,
            "audios": kwargs.get("audios"),
            "spectrograms": kwargs.get("spectrograms"),
            "images": kwargs.get("images"),
            "past_key_values": past_key_values,
            "use_cache": True
        }

if __name__ == "__main__":
    from transformers import HfArgumentParser, WhisperFeatureExtractor, WhisperTokenizer, DonutProcessor
    from model.modeling_whisper import WhisperModel
    import torch
    from collections import OrderedDict
    from dataset.vistext_dataset import VistextDataset, VistextDataCollator
    from transformers import GenerationConfig

    whisper_path = "/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base"
    image_model_path = "/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2"
    ckpt_path = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/output/run_base_donut_10epo/checkpoint-1000.bin"

    whisper_model = WhisperModel.from_pretrained(whisper_path)
    model = DonutWhisper(whisper_model=whisper_model, image_model_path=image_model_path).to(torch.float16)

    ckpt = torch.load(ckpt_path)
    new_ckpt = OrderedDict()
    for k in ckpt.keys():
        new_ckpt[k[len('module.'):]] = ckpt[k]

    kk = model.load_state_dict(new_ckpt, strict=False)
    print(len(kk.unexpected_keys), len(kk.missing_keys))

    tokenizer = WhisperTokenizer.from_pretrained(whisper_path, multilingual=False, language="en", task='transcribe')
    wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
    image_processor = DonutProcessor.from_pretrained(image_model_path)
    
    dataset = VistextDataset("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/test.json", image_processor, wav_processor)
    collate_fn = VistextDataCollator(tokenizer)

    batch = collate_fn([dataset[0]])
    labels = batch.pop("labels")
    audios = batch.pop('audios')
    texts = batch.pop('texts')
    data_ids = batch.pop('data_ids')
    batch["input_ids"] = labels[:, :4]

    model = model.cuda()
    batch['spectrograms'] = batch['spectrograms'].cuda()
    batch['images'] = batch['images'].cuda()
    batch["input_ids"] = batch["input_ids"].cuda()
    
    generation_config = GenerationConfig(max_new_tokens=128, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    
    output = model.generate(generation_config=generation_config, **batch)
    
    preds = [tokenizer.decode(out) for out in output]
    print(preds)