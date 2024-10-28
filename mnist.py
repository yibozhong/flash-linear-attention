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
batch_size, num_heads, seq_len, hidden_size,  = 4, 4, 1024, 1024
device, dtype = 'cuda:0', torch.bfloat16
import time
from transformers import ViTModel, ViTConfig, ViTForImageClassification
# retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)

# student_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
# student_config.num_labels = 10
# student_config.num_hidden_layers = 8
# student_config.image_size = 224 
# model = ViTForImageClassification(student_config).cuda()

# config = RWKV6Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=1, fuse_cross_entropy=False)
# model = RWKV6ForImageClassification(config).to(device=device, dtype=dtype)

# retnet
# config = RetNetConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False)
# model = RetNetForImageClassification(config).to(device=device, dtype=dtype)

# rwkv
# config = RWKV6Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False)
# model = RWKV6ForImageClassification(config).to(device=device, dtype=dtype)

# delta net
# config = DeltaNetConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False, num_heads=8)
# model = DeltaNetForImageClassification(config).to(device=device, dtype=dtype)

# gla
# config = GLAConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False)
# model = GLAForImageClassification(config).to(device=device, dtype=dtype)

# gsa
# config = GSAConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False)
# model = GSAForImageClassification(config).to(device=device, dtype=dtype)

# mamba2
# config = Mamba2Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False, expand=3)
# model = Mamba2ForImageClassification(config).to(device=device, dtype=dtype)

# linear attn
# config = LinearAttentionConfig(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=8, fuse_cross_entropy=False)
# model = LinearAttentionForImageClassification(config).to(device=device, dtype=dtype)

# hgrn2
config = HGRN2Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=4, fuse_cross_entropy=False)
model = HGRN2ForImageClassification(config).to(device=device, dtype=dtype)

# load model from pth
# model.load_state_dict(torch.load('retnet_cifar10_sft_20.pth'))


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# 加载MNIST数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model.requires_grad_(True)
model.train()

# print trainble parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        # print(name)
        pass
print(f"trainable parameters count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 训练模型
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
time1 = time.time()
for epoch in range(20):
    for i, data in enumerate(tqdm(train_dataloader)):
        x, y = data # x: (batch_size, 3, 224, 224), y: (batch_size,)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        y_pred = model(x).logits
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # if i % 100 == 0:
        print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss.item()))
time2 = time.time()
# time in minutes
print((time2 - time1) / 60)
# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_dataloader:
        x, y = data
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        y_pred = model(x).logits
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total)) # 93.96 %

# save the model
torch.save(model.state_dict(), 'hgrn2_cifar10_sft_20.pth')