import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser, WhisperFeatureExtractor, WhisperTokenizer, DonutProcessor, WhisperModel, WhisperForConditionalGeneration
# from model.modeling_whisper import WhisperModel
import transformers
import random
import torch.distributed as dist

from model.donut_whisper import DonutWhisper
from dataset.vistext_dataset import VistextDataset, VistextDataCollator
from train.vistext_trainer import VistextTrainer

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    whisper_path: str = "/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base"
    image_model_path: str = "/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2"
    train_data: str = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_train.json"
    eval_data: str = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_test.json"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = preds == labels
    accuracy = acc[labels != -100].mean()
    return {'accuracy': accuracy}

class AudioProcessor(object):
    def __init__(self, n_mels):
        self.n_mels = n_mels

    def __call__(self, audio):
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio.astype(np.float32), n_mels=self.n_mels)
        return mel

def train():
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # whisper_model = whisper.load_model("base")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(training_args.whisper_path)
    model = DonutWhisper(whisper_model=whisper_model, image_model_path=training_args.image_model_path) # .to(torch.float16)

    image_processor = DonutProcessor.from_pretrained(training_args.image_model_path)

    # wav_processor = AudioProcessor(whisper_model.dims.n_mels)
    wav_processor = WhisperFeatureExtractor.from_pretrained(training_args.whisper_path)

    tokenizer = WhisperTokenizer.from_pretrained(training_args.whisper_path, multilingual=False, language="en", task='transcribe')
    # tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, language="en", task='transcribe') 

    train_dataset = VistextDataset(training_args.train_data, image_processor, wav_processor)
    eval_dataset = VistextDataset(training_args.eval_data, image_processor, wav_processor)
    collate_fn = VistextDataCollator(tokenizer)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    
    trainer = VistextTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics, optimizers=(optimizer, None))

    # for name, param in model.named_parameters():
    #     if  'mix_linear' in name or 'image_linear' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    temp_cnt, temp_total = 0, 0
    if dist.get_rank() == 0:
        for k, p in model.named_parameters():
            temp_total += 1
            if p.requires_grad:
                print(k)
                temp_cnt += 1

        print(temp_cnt, temp_total)

    trainer.train()
    trainer.save_model("final_model")

if __name__ == "__main__":
    train()

# def collate_wrapper(batch):
#     pixel_values = torch.stack([i[0] for i in batch]) 
#     input_features = torch.stack([i[1] for i in batch])
#     labels = [i[2] for i in batch]

#     return pixel_values, input_features, labels

# def logging(s, logfile, logging_=True, log_=True):
#     print(s)
#     if log_:
#         with open(logfile, 'a+') as f_log:
#             f_log.write(s + '\n')
# metrics_wer = evaluate.load("/home/srt11/code_8849/donut_pre_whisper/evaluate/metrics/wer/wer.py")
# metrics_cer = evaluate.load("/home/srt11/code_8849/donut_pre_whisper/evaluate/metrics/cer/cer.py")
# std = EnglishTextNormalizer()
# num_epochs = 100
# batch_size = 32
# lr = 0.00006
# torch.cuda.empty_cache()
# logfile = "/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001_4/logfile.txt"
# logfile_simp = "/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001_4/logfile_simp.txt"
# expdir = "/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001_4"
# load_add = '/home/srt11/data_8849/model/transmodel/train_bz8_lr0.0001_4/model.acc.best.pth'
# scheduler = 'warmuplr'
# accumgrad = 10
# decay_pct = 0.01
# warmup_pct = 0.0
# log_interval = 50
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False, language="en", task='transcribe') 

# model = torch.load(load_add, weights_only=False).to(device)
# # model = transModel(device=device).to(device)
# whisper_model = model.whisper_model

# lora_mode = 999
# lora_pretrain = 0
# lora_alpha = 64
# lora_r = 4 

# ##################
# # Lora
# ##################
# if lora_mode != 999:  # lora_mode=999表示不使用lora
#     import loralib as lora
#     import sys
#     import whisper
#     sys.path.append("/nfs/home/zhaoqiuming/projects/quantization/torch-quantization/code/whisper")
#     from lora_replace import replace_attn_layers, replace_conv_layers, replace_attn_conv_layers, replace_attn_conv_layers_encoder, replace_linear_layers, replace_linear_conv_layers, replace_multi_layers, replace_whisper_layers

#     # replace layer
#     if lora_pretrain == 0:
#         if lora_mode == 0:
#             replace_attn_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 3:
#             replace_conv_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 4:
#             replace_attn_conv_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 7:
#             replace_attn_conv_layers_encoder(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 10:
#             replace_linear_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 14:
#             replace_linear_conv_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 15:
#             replace_whisper_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
#         elif lora_mode == 16:
#             replace_multi_layers(model, lora_alpha=lora_alpha, lora_r=lora_r)
    
#     model.to(device)

#     # freeze
#     for n, p in model.named_parameters():
#         if 'lora_' not in n:
#             p.requires_grad = False
#         else:
#             p.requires_grad = True

# # for name, param in model.named_parameters():
# #     if 'image_conv' in name and 'image_linear' in name and 'image_act' in name:
# #         param.requires_grad = True

# for name, param in model.named_parameters():
#     if  'mix_linear' in name or 'cross_attention' in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# # for name, param in model.named_parameters():
# #     if  'whisper_model.decoder' in name:
# #         param.requires_grad = True
# #     else:
# #         param.requires_grad = False

# summary(model)

# model = nn.DataParallel(model, device_ids=[0,1])

# ###Loading data

# pixel_values_train = '/hdd/srt11/Dataset/train_data/pixel_values'
# input_features_train = '/hdd/srt11/Dataset/train_data/input_features'
# labels_train = '/hdd/srt11/Dataset/train_data/labels_false'

# train_set = CustomDataset(pixel_values_path=pixel_values_train, input_features_path=input_features_train, labels_path=labels_train)
# train_data = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True)

# pixel_values_dev = '/hdd/srt11/Dataset/val_data/pixel_values'
# input_features_dev = '/hdd/srt11/Dataset/val_data/input_features'
# labels_dev = '/hdd/srt11/Dataset/val_data/labels_false'

# dev_set = CustomDataset(pixel_values_path=pixel_values_dev, input_features_path=input_features_dev, labels_path=labels_dev)
# dev_data = DataLoader(dev_set, batch_size=batch_size, collate_fn=collate_wrapper, shuffle=True)
# # dev_data = DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=8)

# options = whisper.decoding.DecodingOptions(language="en", fp16=False, without_timestamps=True)
# special_token_set = set(tokenizer.special_tokens.values())

# criterion = nn.CrossEntropyLoss(ignore_index=-100)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# # decodetask = whisper.decoding.DecodingTask(whisper_model, options)
# # sot_sequence = decodetask.sot_sequence
# # sotlen = len(sot_sequence)





# # trainer
# logging(f"batch_size: {batch_size}", logfile)
# logging(f"batch_size: {batch_size}", logfile_simp)
# # scheduler
# logging(f"scheduler: {scheduler}, lr: {lr}, max_epoch: {num_epochs}", logfile)
# logging(f"scheduler: {scheduler}, lr: {lr}, max_epoch: {num_epochs}", logfile_simp)

# totalwer = 1
# totalcer = 1  
# bestepoch = 0  
# logging("Start of training", logfile)
# logging("Start of training", logfile_simp)
# for epoch in range(num_epochs):
#     start = time.time()
#     totalloss = 0
    
#     #training
#     model.train()
#     with tqdm(train_data, unit="batch") as tepoch:
#         for idx, batch in enumerate(tepoch):
#             pixel_values, input_features, labels = batch
#             pixel_values = to_device(pixel_values, device)
#             input_features = to_device(input_features, device)
#             oritarget = [torch.tensor(y) for y in labels]
#             target = [y[:-1] for y in oritarget]
#             target = pad_sequence(target, batch_first=True, padding_value=tokenizer.eot).to(model.module.device)
#             tepoch.set_description(f"Epoch {epoch + 1}")
#             optimizer.zero_grad()

#             # Forward pass
#             labell = pad_sequence([y[1:] for y in oritarget], batch_first=True, padding_value=-100).to(model.module.device)
#             logits = model(data_donut=pixel_values, data_whisper = input_features, labels=target)
#             # output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)
            
#             loss = criterion(logits.view(-1, logits.size(-1)), labell.view(-1))

#             # Backward and optimize
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             totalloss += loss.item()
#             optimizer.step()
#             tepoch.set_postfix(loss=loss.item())

#             if idx != 0 and (idx + 1) % accumgrad == 0:
#                 # LR scheduler
#                 if scheduler == "warmuplr":
#                     currentstep = epoch * len(train_data) + idx + 1
#                     totalstep = num_epochs * len(train_data)
#                     if currentstep > int(decay_pct * totalstep):
#                         factor = (totalstep - currentstep) / (totalstep - int(decay_pct * totalstep))
#                         optimizer.param_groups[0]['lr'] = lr * max(0, factor)
#                         logging("model-parameter update lr: {}".format(optimizer.param_groups[0]['lr']), logfile)
#                     elif currentstep < int(warmup_pct * totalstep):
#                         factor = currentstep / int(warmup_pct * totalstep)
#                         optimizer.param_groups[0]['lr'] = lr * factor
#                         logging("model-parameter update lr: {}".format(optimizer.param_groups[0]['lr']), logfile)
#                 elif scheduler == "fixlr":
#                     pass
#                 elif scheduler == "steplr":
#                     optimizer.param_groups[0]['lr'] = lr * (0.9 ** epoch)
#                 optimizer.step()
#                 # logging("model-parameter update.", logfile)

#             if idx != 0 and ( idx % log_interval == 0 or idx == 1176 ):
#                 logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
#                     idx, len(train_data), time.time()-start, totalloss/log_interval, 
#                     optimizer.param_groups[0]['lr']), logfile)
#                 logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
#                     idx, len(train_data), time.time()-start, totalloss/log_interval, 
#                     optimizer.param_groups[0]['lr']), logfile_simp)
#                 totalloss = 0

#     # validation
#     model.eval()
#     totalloss = 0
#     o_list, l_list = [], []
#     WER = 0
#     CER = 0
#     with torch.no_grad():
#         for idx, (pixel_values, input_features, labels) in enumerate(tqdm(dev_data, unit="batch")):          
#             pixel_values = to_device(pixel_values, device)
#             input_features = to_device(input_features, device)
#             oritarget = [torch.tensor(y) for y in labels]
#             labell = pad_sequence([y[1:] for y in oritarget], batch_first=True, padding_value=-100).to(model.module.device)
#             target = pad_sequence([y[:-1] for y in oritarget], batch_first=True, padding_value=tokenizer.eot).to(model.module.device)
            
#             # Forward pass
#             logits = model(data_donut=pixel_values, data_whisper = input_features, labels=target)
#             loss = criterion(logits.view(-1, logits.size(-1)), labell.view(-1))
#             totalloss += loss.item()
#             # logits[logits == -100] = tokenizer.eot
#             labell[labell == -100] = tokenizer.eot
     
#             # output = torch.log_softmax(logits[:, sotlen-1:-1], dim=-1)

#             for o, l in zip(logits, labels):
#                 o = torch.argmax(o, dim=1)
#                 o[o == -100] = tokenizer.eot
#                 o_out = std(tokenizer.decode([t for t in o if t not in special_token_set]))
#                 l_out = std(tokenizer.decode([t for t in l if t not in special_token_set]))
#                 if not l_out.strip():
#                     l_out = '<empty>'
#                 if not o_out.strip():
#                     o_out = '<empty>'
#                 o_list.append(o_out)
#                 l_list.append(l_out)

#             if idx % 20 == 0 and idx > 0:
#                 cer = metrics_cer.compute(references=l_list, predictions=o_list)
#                 wer = metrics_wer.compute(references=l_list, predictions=o_list)
#                 logging("{} out of {} finished | WER: {} | CER:{} | Loss: {}".format(
#                     idx, len(dev_data), wer, cer, totalloss/20), logfile)
#                 totalloss = 0
#                 WER = wer
#                 CER = cer
#             if WER > 0.1:
#                 break
                
#         logging("[epoch {}] Total WER: {}, CER: {}".format(epoch+1, WER, CER), logfile)
#         logging("[epoch {}] Total WER: {}, CER: {}".format(epoch+1, WER, CER), logfile_simp)

#         if WER < totalwer:
#             torch.save(model.module, os.path.join(expdir, "model.acc.best.pth"))
#             totalwer = WER
#             bestepoch = epoch + 1
#             logging("Saving best model at epoch {}".format(epoch+1), logfile)
#             logging("Saving best model at epoch {}".format(epoch+1), logfile_simp)

#         torch.save(model.module, os.path.join(expdir, "snapshot.ep.{}.pth".format(epoch+1)))

#         with open(f"out2/{epoch+1}_out.txt", "w", encoding="utf-8") as file:
#             for r in o_list:
#                 file.write(r)
#                 file.write("\n")
#         with open(f"out2/{epoch+1}_ref.txt", "w", encoding="utf-8") as file:
#             for r in l_list:
#                 file.write(r)
#                 file.write("\n")

# logging("Saving best model at epoch {} , Max ACC: {}".format(bestepoch, totalwer), logfile)
# logging("Saving best model at epoch {} , Max ACC: {}".format(bestepoch, totalwer), logfile_simp)

# print("Program Over")