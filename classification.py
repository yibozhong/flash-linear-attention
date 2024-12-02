import os
# if you are in china, use this mirror instead, otherwise you can comment out this line
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
from fla.models import ABCConfig, ABCForImageClassification
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
from transformers import Swinv2ForImageClassification
from mambavision import create_model
device, dtype = torch.device('cuda'), torch.float32
import torch.profiler as profiler
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

# import amp and scaler
from torch.amp import GradScaler, autocast

# Setup logging
def setup_logging(args):
    log_filename = f'training_{args.model}_{args.dataset}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_filename}")

# to get a understanding of the efficiency of the model, we can use the torch profiler
def train_with_profiler(model, train_dataloader, criterion, optimizer, device, dtype, args, scheduler=None):
    if args.amp_enabled:
        scaler = GradScaler()
    model.train()
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=profiler.tensorboard_trace_handler(args.log_dir), # specify this directory in the args
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(args.epochs):
            for i, data in enumerate(tqdm(train_dataloader)):
                x, y = data  # x: (batch_size, 3, 224, 224), y: (batch_size,)
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device)
                if args.amp_enabled:
                    with autocast(device_type=device):
                        y_pred = model(x).logits
                        loss = criterion(y_pred, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    y_pred = model(x).logits
                    loss = criterion(y_pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                prof.step() 
            if scheduler is not None:
                scheduler.step()

def eval(model, dataloader, args):
    model.eval()
    correct = 0
    total = 0
    
    # Use CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for data in tqdm(dataloader):
            x, y = data
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device)
            
            if args.amp_enabled:
                with autocast(device_type='cuda'):
                    y_pred = model(x).logits
            else:
                y_pred = model(x).logits
                
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    acc = correct / total    
    end_event.record()
    torch.cuda.synchronize()
    eval_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    logging.info(f"Evaluation time: {eval_time / 60:.2f} minutes")
    return acc  


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Model Selection')
    # model name
    parser.add_argument('--model', type=str, required=True, help='Model name, for example')
    # dataset name
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset name')
    # numebr of hidden layers
    parser.add_argument('--num_hidden_layers', type=int, default=7, help='Number of hidden layers')
    # number of labels
    parser.add_argument('--num_labels', type=int, default=10, help='Number of labels')
    # hidden size
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size')
    # patch size
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    # resolution
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    # epochs
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    # whether to enable amp
    parser.add_argument('--amp_enabled', action='store_true', help='Enable AMP for faster training')
    # eval epoch
    parser.add_argument('--eval_epoch', type=int, default=10, help='Number of epochs for eval')
    # Add learning rate arguments
    parser.add_argument('--b_lr', type=float, default=2e-4, help='Learning rate for backbone')
    parser.add_argument('--h_lr', type=float, default=2e-4, help='Learning rate for head')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')
    # batch size
    parser.add_argument('--train_bs', type=int, default=128, help='Training batch size')
    parser.add_argument('--eval_bs', type=int, default=256, help='Evaluation batch size')
    # number of dataloader workers
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')
    # number of heads
    parser.add_argument('--num_heads', type=int, default=12, help='Number of heads')
    # log dir
    parser.add_argument('--log_dir', type=str, default='./log', help='Tensorboard log directory')
    # chunk size
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size')
    # expand k
    parser.add_argument('--expand_k', type=float, default=1.0, help='Expand k')
    # expand v
    parser.add_argument('--expand_v', type=float, default=1.0, help='Expand v')
    # seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')


    return parser.parse_args()


def get_model(args):
    if args.model == 'vit':
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        config.num_labels = args.num_labels
        config.num_hidden_layers = args.num_hidden_layers # change to 6 when ps=2
        config.image_size = args.image_size
        config.patch_size = args.patch_size
        config.hidden_size = args.hidden_size
        config.num_attention_heads = args.num_heads 
        model = ViTForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'mambavision':
        model = create_model('mamba_vision_S').to(device=device, dtype=dtype)
    elif args.model == 'abc':
        config = ABCConfig(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='gelu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='fused_chunk',
            num_heads=args.num_heads
        )
        model = ABCForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'retnet':
        config = RetNetConfig(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='fused_chunk',
            num_heads=args.num_heads
        )
        model = RetNetForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'rwkv':
        config = RWKV6Config(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='chunk',
            num_heads=args.num_heads
        )
        model = RWKV6ForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'deltanet':
        config = DeltaNetConfig(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            num_heads=args.num_heads,
            attn_mode='fused_chunk'
        )
        model = DeltaNetForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'gla':
        config = GLAConfig(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='relu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='fused_chunk',
            num_heads=args.num_heads
        )
        config.conv = True
        model = GLAForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'swin':
        model = Swinv2ForImageClassification.from_pretrained('microsoft/swinv2-tiny-patch4-window8-224').to(device=device, dtype=dtype)
    elif args.model == 'gsa':
        config = GSAConfig(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=736,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='fused_chunk',
            num_heads=args.num_heads
        )
        model = GSAForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'mamba2':
        config = Mamba2Config(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            expand=3
        )
        model = Mamba2ForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'linear_attn':
        config = LinearAttentionConfig(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='chunk'
        )
        model = LinearAttentionForImageClassification(config).to(device=device, dtype=dtype)
    elif args.model == 'hgrn2':
        config = HGRN2Config(
            vision=True,
            class_size=args.num_labels,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_size=1024,
            hidden_act='silu',
            num_hidden_layers=args.num_hidden_layers,
            fuse_cross_entropy=False,
            attn_mode='fused_chunk',
            num_heads=args.num_heads
        )
        model = HGRN2ForImageClassification(config).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model, config

def get_data(args):
    transform_for_rgb_images = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(), 
    ])
    transform_for_grayscale_images = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((args.image_size, args.image_size)), 
        transforms.ToTensor(), 
    ])
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_for_rgb_images)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_for_rgb_images)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_for_rgb_images)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_for_rgb_images)
    elif args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_for_grayscale_images)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_for_grayscale_images)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_bs, shuffle=False, num_workers=args.num_workers)

    return train_dataloader, test_dataloader

def train_one_epoch(model, train_dataloader, criterion, optimizer, scheduler, device, dtype, args, epoch_num=0):
    model.train()
    total_loss = 0
    if args.amp_enabled:
        scaler = GradScaler()
    
    for i, (x, y) in enumerate(tqdm(train_dataloader)):
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device)
        
        if args.amp_enabled:
            with autocast(device_type='cuda'):
                y_pred = model(x).logits
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            y_pred = model(x).logits
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        
        if i % 20 == 0:
            print(f'Epoch {epoch_num}, Batch {i}, Loss: {loss.item():.4f}')
            logging.info(f'Epoch {epoch_num}, Batch {i}, Loss: {loss.item():.4f}')
    
    scheduler.step()
    return total_loss / len(train_dataloader)

def get_param_groups(model, args):
    """Helper function to separate parameters into backbone and head groups"""
    backbone_params = []
    head_params = []
    
    # Check if model has standard attribute structure
    if hasattr(model, 'model') and hasattr(model, 'vm_head'):
        backbone_params.extend(model.model.parameters())
        head_params.extend(model.vm_head.parameters())
    else:
        # Fallback: group parameters by name
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(x in name.lower() for x in ['head', 'classifier', 'vm_head']):
                    head_params.append(param)
                else:
                    backbone_params.append(param)

    
    return [
        {'params': backbone_params, 'lr': args.b_lr},
        {'params': head_params, 'lr': args.h_lr}
    ]

def main():
    args = get_args()
    
    # Setup logging first, before any logging calls
    setup_logging(args)
    
    # Log training parameters
    logging.info("================== Training Setup ==================")
    logging.info(f"Training parameters:")
    logging.info(f"Model: {args.model}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Hidden size: {args.hidden_size}")
    logging.info(f"Number of heads: {args.num_heads}")
    logging.info(f"Number of layers: {args.num_hidden_layers}")
    logging.info(f"Batch size: {args.train_bs}")
    logging.info(f"Learning rates - backbone: {args.b_lr}, head: {args.h_lr}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {args.seed}")
    
    # Get model and config
    model, config = get_model(args)
    
    # Log model information
    logging.info("================== Model Info ==================")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model configuration: {config}")
    logging.info(f"Trainable parameters count: {trainable_params:,}")
    
    model.requires_grad_(True)
    print(f"trainable parameters count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # log the model configuration
    logging.info(f"Model configuration: {config}")
    # log the parameters
    logging.info(f"trainable parameters count of {args.model}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataloader, test_dataloader = get_data(args)  
    criterion = torch.nn.CrossEntropyLoss()
    
    # Split parameters into two groups and set different learning rates
    param_groups = get_param_groups(model, args)
    
    optimizer = optim.AdamW(param_groups, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0
    best_acc_epoch = 0
    
    # record the start time and end time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        avg_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, 
            scheduler, device, dtype, args, epoch_num=epoch
        )
        if epoch % args.eval_epoch == 0:
            acc = eval(model, test_dataloader, args)
            logging.info(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc*100:.2f}%')
        
        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
            }, f"{args.model}_best.pth")
            
            logging.info(f"New best model saved with accuracy: {acc*100:.2f}%")
    
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    logging.info(f"Training completed in {total_time / 60.0:.2f} minutes")
    logging.info(f"Best Accuracy: {best_acc*100:.2f}% at epoch {best_acc_epoch}")

if __name__ == '__main__':
    main()