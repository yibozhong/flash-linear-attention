from transformers import pipeline
from PIL import Image
import requests
from datasets import load_dataset
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict, Image
from transformers import AutoImageProcessor
from torchvision.transforms import ColorJitter
import numpy as np
import torch
from torch import nn
from transformers import TrainingArguments, Trainer
from segformer import SegformerForSemanticSegmentation
import evaluate
import time

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics

# dataset
ds = load_dataset("scene_parse_150", split="train", cache_dir="./data")
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]
# id and labels
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
# preprocess
checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def train_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs

train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


print(train_ds[0]['pixel_values'].shape)
print(train_ds[0]['labels'].shape)
print(test_ds[0]['pixel_values'].shape)
print(test_ds[0]['labels'].shape)

metric = evaluate.load("mean_iou")

# model

model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)


# check trainbale parameters

param_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        param_count += param.numel()

print(f"total parameters : {param_count}")

print(train_ds[0]['pixel_values'].shape)


training_args = TrainingArguments(
    output_dir="segformer-b0-scene-parse-150",
    learning_rate=1e-4,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_total_limit=1,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=1,
    logging_steps=50,
    eval_accumulation_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)

time1 = time.time()

trainer.train()

time2 = time.time()

print(f"training time : {(time2 - time1) / 60}")

eval_results = trainer.evaluate()

# 打印评估结果
print("Evaluation results:", eval_results)
# write the results to a file
with open("eval_results.txt", "w") as f:
    f.write(json.dumps(eval_results))