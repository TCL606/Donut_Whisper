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
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser, WhisperFeatureExtractor, WhisperTokenizer, DonutProcessor
from model.modeling_whisper import WhisperModel
import transformers
import random
import torch.distributed as dist

from transformers import GenerationConfig
from model.donut_whisper import DonutWhisper
from dataset.vistext_dataset import VistextDataset, VistextDataCollator
from train.vistext_trainer import VistextTrainer

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    whisper_path: str = None
    image_model_path: str = None
    train_data: str = None
    eval_data: str = None
    test_data: str = None
    do_test: bool = False
    ckpt: str = None

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    acc = preds == labels
    accuracy = acc[labels != -100].mean()
    return {'accuracy': accuracy}

def train():
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    if training_args.deepspeed == "None":
        training_args.deepspeed = None

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    whisper_model = WhisperModel.from_pretrained(training_args.whisper_path)
    model = DonutWhisper(whisper_model=whisper_model, image_model_path=training_args.image_model_path).to(torch.float16)

    image_processor = DonutProcessor.from_pretrained(training_args.image_model_path)
    wav_processor = WhisperFeatureExtractor.from_pretrained(training_args.whisper_path)
    tokenizer = WhisperTokenizer.from_pretrained(training_args.whisper_path, multilingual=False, language="en", task='transcribe')

    if not training_args.do_test:
        train_dataset = VistextDataset(training_args.train_data, image_processor, wav_processor)
        eval_dataset = VistextDataset(training_args.eval_data, image_processor, wav_processor)
        collate_fn = VistextDataCollator(tokenizer)

        if training_args.deepspeed is not None:
            trainer = VistextTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics) # , optimizers=(optimizer, None)
        else:
            pass
            # optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
            # trainer = VistextTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=collate_fn, compute_metrics=compute_metrics, optimizers=(optimizer, None))

        for name, param in model.named_parameters():
            if  'mix_linear' in name or 'image_linear' in name or 'decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

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

    else:
        ckpt = torch.load(training_args.ckpt)
        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)
        print(len(kk.unexpected_keys), len(kk.missing_keys))

        test_dataset = VistextDataset(training_args.test_data, image_processor, wav_processor)
        collate_fn = VistextDataCollator(tokenizer)
        trainer = VistextTrainer(model=model, args=training_args, train_dataset=test_dataset, eval_dataset=test_dataset, data_collator=collate_fn, compute_metrics=compute_metrics)
        
        generation_config = GenerationConfig(max_new_tokens=128, do_sample=False, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        outputs = trainer.predict(test_dataset, generation_config=generation_config)
        for item in outputs:
            if item["pred"] is not None:
                item["pred"] = tokenizer.decode(item["pred"])

        if dist.get_rank() == 0:
            os.makedirs(os.path.join(training_args.output_dir, "test"), exist_ok=True)

        dist.barrier()
        with open(os.path.join(training_args.output_dir, "test", f"results_{dist.get_rank()}.json"), 'w') as fp:
            json.dump(outputs, fp)

        dist.barrier()

        if dist.get_rank() == 0:
            res = []
            print("Start Merging")
            for i in range(dist.get_world_size()):
                with open(os.path.join(training_args.output_dir, "test", f"results_{i}.json"), 'r') as fp:
                    data_i = json.load(fp)
                res += data_i
            with open(os.path.join(training_args.output_dir, "test", f"results_final.json"), 'w') as fp:
                json.dump(res, fp, indent=4)
            print(os.path.join(training_args.output_dir, "test", f"results_final.json"))

        dist.barrier()

if __name__ == "__main__":
    train()
