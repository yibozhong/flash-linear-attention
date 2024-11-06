from transformers import pipeline
from PIL import Image
import requests
from datasets import load_dataset
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import Dataset, DatasetDict
from PIL import Image
from torchvision.transforms import ColorJitter
import numpy as np
import torch
from torch import nn
from transformers import TrainingArguments, Trainer
from segformer import SegformerForSemanticSegmentation
from transformers import AutoImageProcessor
import evaluate
import time
import json
from huggingface_hub import hf_hub_url
from huggingface_hub import hf_hub_download

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",    
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=image_processor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

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

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)


def train_transforms(example_batch):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
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

model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)


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
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=50,
    label_names=["labels"],
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