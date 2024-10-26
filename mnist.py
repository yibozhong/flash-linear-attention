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
from fla.models import RWKV6ForImageClassification, RWKV6Config
batch_size, num_heads, seq_len, hidden_size,  = 4, 4, 1024, 1024
device, dtype = 'cuda:0', torch.bfloat16
# retnet = MultiScaleRetention(hidden_size=hidden_size, num_heads=num_heads).to(device=device, dtype=dtype)


config = RWKV6Config(vision=True, class_size=10, image_size=224, patch_size=16, hidden_size=768, hidden_act='silu', num_hidden_layers=1, fuse_cross_entropy=False)
model = RWKV6ForImageClassification(config).to(device=device, dtype=dtype)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
    # transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model.requires_grad_(True)
model.train()

# print trainble parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
print(f"trainable parameters count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 训练模型
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(1):
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
        if i % 100 == 0:
            print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, loss.item()))

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

