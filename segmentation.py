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
from torch.utils.data import Dataset
import numpy as np
from torch.optim import Adam
import monai
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs
  
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader):
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        print(f"Loss: {loss.item()}")
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


# load dataset
ds = load_dataset("nielsr/breast-cancer", split="train")
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]


# define the models
model_name = "facebook/sam-vit-base"
config = SamConfig.from_pretrained(model_name)
model = SamModel(config).to(device)
model.requires_grad_(True)
model.train()
# change hidden size

processor = SamProcessor.from_pretrained(model_name)

train_ds = SAMDataset(train_ds, processor)
test_ds = SAMDataset(test_ds, processor)

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

