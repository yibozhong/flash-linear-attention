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
from segformer import SegformerForSemanticSegmentation, SegformerConfig, SegGLAForSemanticSegmentation
from transformers import AutoImageProcessor
import evaluate
import time
import json
from huggingface_hub import hf_hub_url
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from fla.models import GLAConfig

# fix seed to 42
torch.manual_seed(42)

class SegTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        num_examples = len(eval_dataloader)
        total_loss = 0.0
        # Initialize metrics
        total_metrics = None
        num_batches = 0
        
        for inputs in tqdm(eval_dataloader):
            # Forward pass
            with torch.no_grad():
                losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys)
                # Scale the logits to the size of the label
                logits = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1)
                
                # Convert to numpy
                pred_labels = logits.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                
                # Compute metrics for the batch
                batch_metrics = metric._compute(
                    predictions=pred_labels,
                    references=labels,
                    num_labels=len(id2label),
                    ignore_index=0,
                    reduce_labels=image_processor.do_reduce_labels,
                )
                
                # Delete intermediate variables
                # del logits
                # del labels
                # del pred_labels
                
                # Accumulate metrics
                if total_metrics is None:
                    total_metrics = batch_metrics
                else:
                    for key in total_metrics:
                        total_metrics[key] += batch_metrics[key]
                 
                total_loss += losses.sum().item()    
                # Manually empty the cache
                torch.cuda.empty_cache()
                num_batches += 1
        
        # Average the metrics
        # for key in total_metrics:
        #     total_metrics[key] /= num_batches
        total_loss /= num_batches
        # Remove unwanted metrics
        total_metrics.pop("per_category_accuracy", None)
        total_metrics.pop("per_category_iou", None)
        # add new key loss to metric
        # print(f"\ntotal eval average loss: {total_loss}")
        total_metrics["eval_loss"] = total_loss
        # create a field named eval_loss in the metrics

        # log
        self.log(total_metrics)
        
        return total_metrics

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        print(logits_tensor.shape, labels.shape)
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

        # delete these two fields
        metrics.pop("per_category_accuracy")
        metrics.pop("per_category_iou")

        # metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        # metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred.predictions, eval_pred.label_ids
#     batch_size = 32
#     # Convert logits to tensor
#     logits_tensor = torch.from_numpy(logits)
    
#     # Initialize metrics
#     total_metrics = None
    
#     # Process in batches
#     for i in range(0, logits_tensor.shape[0], batch_size):
#         batch_logits = logits_tensor[i:i + batch_size]
#         batch_labels = labels[i:i + batch_size]
#         print(batch_logits.shape, batch_labels.shape)
#         # Scale the logits to the size of the label
#         batch_logits = nn.functional.interpolate(
#             batch_logits,
#             size=batch_labels.shape[-2:],
#             mode="bilinear",
#             align_corners=False,
#         ).argmax(dim=1)
        
#         # Convert to numpy
#         batch_pred_labels = batch_logits.detach().cpu().numpy()
        
#         # Compute metrics for the batch
#         batch_metrics = metric._compute(
#             predictions=batch_pred_labels,
#             references=batch_labels,
#             num_labels=len(id2label),
#             ignore_index=0,
#             reduce_labels=image_processor.do_reduce_labels,
#         )
        
#         del batch_logits
#         del batch_labels
#         del batch_pred_labels

#         # Accumulate metrics
#         if total_metrics is None:
#             total_metrics = batch_metrics
#         else:
#             for key in total_metrics:
#                 total_metrics[key] += batch_metrics[key]
        
#         torch.cuda.empty_cache()
    
#     # Average the metrics
#     for key in total_metrics:
#         total_metrics[key] /= (logits_tensor.shape[0] / batch_size)
    
#     # Remove unwanted metrics
#     total_metrics.pop("per_category_accuracy", None)
#     total_metrics.pop("per_category_iou", None)
    
#     return total_metrics

# dataset
ds = load_dataset("scene_parse_150", split="train[:5000]", cache_dir="./data")
ds = ds.train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
# train_ds = load_dataset("scene_parse_150", split="train", cache_dir="./data")
# # test_ds = load_dataset("scene_parse_150", split="validation", cache_dir="./data")
# print(len(train_ds), len(test_ds))

# id and labels
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
# preprocess
checkpoint = "nvidia/mit-b1"
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
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
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

metric = evaluate.load("mean_iou")

config = SegformerConfig.from_pretrained(checkpoint)
# model
# print(config.num_labels)

# config_gla = GLAConfig(vision=True, attn_mode='fused_chunk', num_heads=2)
# # combine the two config
# config_dict = config_gla.to_dict()
# config_dict.update(config.to_dict())
# merged_config = SegformerConfig.from_dict(config_dict)
# merged_config.id2label = id2label
# merged_config.label2id = label2id
# merged_config.ignore_mismatched_sizes = True
# # print(merged_config.semantic_loss_ignore_index)
# print(merged_config.num_labels)
# # model = SegformerForSemanticSegmentation(merged_config)
# model = SegGLAForSemanticSegmentation(merged_config)
# model = SegformerForSemanticSegmentation(merged_config)
model = SegformerForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

# print(model.config.num_labels)
# # difference between config and model
# for k, v in model.config.to_dict().items():
#     if k in config.to_dict():
#         if v != config.to_dict()[k]:
#             print(f"diff in {k}")
#     else:
#         # print(k, v)
#         pass

# check trainbale parameters
model.requires_grad_(True)
param_count = 0
for name, param in model.named_parameters():
    # if param.requires_grad:
    param_count += param.numel()

print(f"total parameters : {param_count}")

# print(train_ds[0]['pixel_values'].shape)


training_args = TrainingArguments(
    output_dir="segformer-b0-scene-parse-150",
    learning_rate=1e-3,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_total_limit=1,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1,
    label_names=["labels"],
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    # dataloader_num_workers=4,
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