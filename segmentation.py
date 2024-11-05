from datasets import load_dataset
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor
from sam import SamModel, SamProcessor, SamConfig
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import numpy as np
# load dataset
ds = load_dataset("scene_parse_150", split="train[:50]", cache_dir="./data/")
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        images = batch['pixel_values'].to(device)
        labels = batch['label'].to(device).long()

        optimizer.zero_grad()
        masks_pred = model(images).pred_masks
        loss = criterion(masks_pred, labels)
        loss.backward()
        optimizer.step()
        print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, metric):
    model.eval()
    total_iou = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            masks = outputs.logits

            # 计算IoU
            iou = metric.compute(predictions=masks.argmax(dim=1).cpu().numpy(), references=labels.cpu().numpy())
            total_iou += iou['mean_iou']
            total_samples += 1
    mean_iou = total_iou / total_samples
    return mean_iou


def collate_fn(batch):
    pass

# define label mapping
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# define the models
model_name = "facebook/sam-vit-base"
config = SamConfig.from_pretrained(model_name)
model = SamModel(config).to(device)
model.requires_grad_(True)
model.train()
# change hidden size

processor = SamProcessor.from_pretrained(model_name)

def transforms(example_batch):
    images = example_batch["image"]
    labels = example_batch["annotation"]
    inputs = processor(images, return_tensors="pt", )
    labels = processor(images, return_tensors="pt", )
    a = inputs["pixel_values"]
    b = labels['pixel_values']
    # resize b using torchvision
    # print(type(a), type(b))
    return {"pixel_values": a, "label": b}


train_ds.set_transform(transforms,)
test_ds.set_transform(transforms,)

metric = evaluate.load("mean_iou")

# train the model
train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_dataloader, optimizer, device, criterion)
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
    test_iou = evaluate_model(model, test_dataloader, device, metric)
    print(f"Epoch {epoch + 1}, Test IoU: {test_iou:.4f}")

