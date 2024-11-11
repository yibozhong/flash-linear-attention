import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from tqdm import tqdm
import sys
import math
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fla.layers import MultiScaleRetention
from fla.models import RWKV6ForImageClassification, RWKV6Config, RetNetConfig, RetNetForImageClassification
from fla.models import DeltaNetConfig, DeltaNetForImageClassification
from fla.models import GLAConfig, GLAForImageClassification
from fla.models import GSAConfig, GSAForImageClassification
from fla.models import Mamba2Config, Mamba2ForImageClassification
from fla.models import LinearAttentionConfig, LinearAttentionForImageClassification
from fla.models import HGRN2Config, HGRN2ForImageClassification
import time
from transformers import ViTModel, ViTConfig, ViTForImageClassification
from transformers import AutoModelForImageClassification
from mambavision import create_model
device, dtype = torch.device('cuda'), torch.bfloat16
import torch.profiler as profiler

def train_with_profiler(model, train_dataloader, criterion, optimizer, device, dtype):
    model.train()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(1):
            for i, data in enumerate(tqdm(train_dataloader)):
                x, y = data  # x: (batch_size, 3, 224, 224), y: (batch_size,)
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                y_pred = model(x).logits
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prof.step()  # 记录当前步骤的性能数据
                if i % 100 == 0:
                    print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss.item()))
                if i >= 100:
                    break

def eval(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    time1 = time.time()
    with torch.no_grad():
        for data in tqdm(dataloader):
            x, y = data # x: (batch_size, 3, 224, 224), y: (batch_size,)
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device)
            y_pred = model(x).logits
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct / total
    time2 = time.time()
    print(f"eval time : {(time2 - time1) / 60}")
    return acc  


# retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)

# student_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
# student_config.num_labels = 10
# student_config.num_hidden_layers = 7 # change to 6 when ps=2
# student_config.image_size = 224
# student_config.patch_size = 2
# student_config.hidden_size = 768
# model = ViTForImageClassification(student_config)
# model = model.to(device=device, dtype=dtype)

# mambavision

# model = create_model('mamba_vision_S').to(device=device, dtype=dtype)
# model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-S-1K", trust_remote_code=True).to(device=device, dtype=dtype)

# config = RWKV6Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=1, fuse_cross_entropy=False)
# model = RWKV6ForImageClassification(config).to(device=device, dtype=dtype)

# retnet
# config = RetNetConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=7, fuse_cross_entropy=False, attn_mode='fused_chunk', num_heads=12)
# model = RetNetForImageClassification(config).to(device=device, dtype=dtype)

# rwkv
# config = RWKV6Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=7, fuse_cross_entropy=False, attn_mode='chunk', num_heads=12)
# model = RWKV6ForImageClassification(config).to(device=device, dtype=dtype)

# delta net
# config = DeltaNetConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=7, fuse_cross_entropy=False, num_heads=12, attn_mode='fused_chunk')
# model = DeltaNetForImageClassification(config).to(device=device, dtype=dtype)

# gla
config = GLAConfig(vision=True, class_size=100, image_size=14, patch_size=16, hidden_size=768, hidden_act='relu', num_hidden_layers=4, fuse_cross_entropy=False, attn_mode='fused_chunk', num_heads=12)
config.conv = True
model = GLAForImageClassification(config).to(device=device, dtype=dtype)


# gsa
# config = GSAConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=736, hidden_act='silu', num_hidden_layers=7, fuse_cross_entropy=False, attn_mode='fused_chunk', num_heads=8)
# model = GSAForImageClassification(config).to(device=device, dtype=dtype)

# mamba2
# config = Mamba2Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False, expand=3)
# model = Mamba2ForImageClassification(config).to(device=device, dtype=dtype)

# linear attn
# config = LinearAttentionConfig(vision=True, class_size=10, image_size=224, patch_size=2, hidden_size=768, hidden_act='silu', num_hidden_layers=7, fuse_cross_entropy=False, attn_mode='chunk')
# model = LinearAttentionForImageClassification(config).to(device=device, dtype=dtype)

# hgrn2
# config = HGRN2Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=1024, hidden_act='silu', num_hidden_layers=4, fuse_cross_entropy=False, attn_mode='fused_chunk', num_heads=8)
# model = HGRN2ForImageClassification(config).to(device=device, dtype=dtype)

# load model from pth
# model.load_state_dict(torch.load('retnet_cifar10_sft_20.pth'))

# model.load_state_dict(torch.load('gsa_cifar100_sft_res32_ps1_30epoch_with_aug_layer15_hs336_22M_params.pth'))

transform = transforms.Compose([
    transforms.Resize((227, 227)), 
    transforms.ToTensor(), 
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)

test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

model.requires_grad_(True)
model.train()
conv_param = 0
# print trainble parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        # print(name)
        if 'conv' in name or 'embed' in name:
            print(name)
            print(param.numel())
            conv_param += param.numel()
print(f"conv parameters count: {conv_param}")
print(f"trainable parameters count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
best_acc = 0
best_acc_epoch = 0
time1 = time.time()
for epoch in range(100):
    model.train()
    for i, data in enumerate(tqdm(train_dataloader)):
        x, y = data # x: (batch_size, 3, 224, 224), y: (batch_size,)
        # print(x.shape)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device)
        y_pred = model(x).logits
        # y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # clip gradient
        # if i % 100 == 0:
        print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss.item()))

    acc = eval(model, test_dataloader)
    if acc > best_acc:
        best_acc = acc
        best_acc_epoch = epoch + 1
        torch.save(model.state_dict(), f"convgla_cifar100_sft.pth")
    print('Accuracy of the model on the test images: {} %'.format(100 * acc))

print(f"Best Accuracy: {best_acc}")
print(f"Best Accuracy Epoch: {best_acc_epoch}")

# train_with_profiler(model, train_dataloader, criterion, optimizer, device, dtype)

model.requires_grad_(False)
model.eval()
acc = eval(model, test_dataloader)
time2 = time.time()
# time in minutes
print(f"training time : {(time2 - time1) / 60.0} minutes")