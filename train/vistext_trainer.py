from transformers import Trainer
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict
import torch.nn as nn

class VistextTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        logits, _ = model(**inputs)
        logits = logits[:, 3:-1, :] # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>

        targets = inputs["labels"]
        targets = targets[:, 4:]

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        outputs = {"loss": loss, "logits": logits}

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys = None,
    ):
        logits, _ = self.model(**inputs)
        logits = logits[:, 3:-1, :] # <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
        labels = inputs["labels"]
        labels = labels[:, 4:]
        if prediction_loss_only:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return loss, logits, labels
        else:
            return None, logits, labels