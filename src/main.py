import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import json
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import (DecoderOnlyLanguageModel, CausalDecoderBlock,
                   NoResidualCausalDecoderBlock)  # (model.py 中其他的类是 model.py 内部使用的，这里可以不导入)
from data_utils import load_data, get_batch
from utils import set_seed, generate_causal_mask

# --- (修正点：将 PAD_TOKEN 定义在全局范围) ---
PAD_TOKEN = -1


# --- (训练与评估函数) ---

def train_one_epoch(model, optimizer, device, train_data, bptt, vocab_size):
    model.train()
    total_loss = 0.
    num_batches = 0
    for i in range(0, train_data.size(0) - 1, bptt):
        data, targets = get_batch(train_data, i, bptt)
        seq_len = data.size(1)
        mask = generate_causal_mask(seq_len, device)

        optimizer.zero_grad()
        output = model(data, mask)

        loss = F.cross_entropy(output.view(-1, vocab_size),
                               targets.view(-1),
                               ignore_index=PAD_TOKEN)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


def evaluate_model(model, device, val_data, bptt, vocab_size):
    model.eval()
    total_loss = 0.
    num_batches = 0
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, bptt):
            data, targets = get_batch(val_data, i, bptt)
            seq_len = data.size(1)
            mask = generate_causal_mask(seq_len, device)
            output = model(data, mask)

            # (现在可以正确访问 PAD_TOKEN)
            loss = F.cross_entropy(output.view(-1, vocab_size),
                                   targets.view(-1),
                                   ignore_index=PAD_TOKEN)

            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches


# --- (实验运行函数) ---
def run_experiment(model_instance, optimizer, scheduler, device, num_epochs,
                   train_data, val_data, bptt, vocab_size,
                   experiment_name="Baseline"):
    print(f"\n--- 运行实验: {experiment_name} ---")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        avg_train_loss = train_one_epoch(model_instance, optimizer, device, train_data, bptt, vocab_size)
        train_losses.append(avg_train_loss)
        avg_val_loss = evaluate_model(model_instance, device, val_data, bptt, vocab_size)
        val_losses.append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('results', exist_ok=True)
            model_save_path = f'results/{experiment_name}_best_model.pth'
            torch.save(model_instance.state_dict(), model_save_path)
            print(f"模型改进，已保存至: {model_save_path}")

        scheduler.step()

    print(f"--- 实验 {experiment_name} 结束. ---")

    metrics = {
        "experiment_name": experiment_name,
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    with open(f'results/{experiment_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"指标已保存至: results/{experiment_name}_metrics.json")

    return train_losses, val_losses


# --- (主入口函数) ---
def main():
    parser = argparse.ArgumentParser(description="Transformer Ablation Study")
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['baseline', 'no_pe', 'no_res', 'single_head'],
                        help='Which experiment to run.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--bptt', type=int, default=35, help='BPTT sequence length.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW.')
    args = parser.parse_args()

    # 1. 设置可复现性
    set_seed(args.seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 2. 设置设备和加载数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_data, val_data, VOCAB_SIZE = load_data(device, args.batch_size, args.batch_size)
    except FileNotFoundError:
        # 如果 load_data 抛出异常 (找不到 input.txt)，则程序在此处停止
        return

    # 3. 定义固定的模型超参数
    D_MODEL = 64
    N_LAYERS = 2
    N_HEADS_BASELINE = 4  # 基准头数
    D_FF = 256
    DROPOUT = 0.1

    print(f"\n--- 初始化实验: {args.experiment} ---")
    print(f"Seed: {args.seed}, Epochs: {args.epochs}, LR: {args.lr}")

    # 4. 根据实验名称构建模型
    n_heads = N_HEADS_BASELINE

    if args.experiment == 'single_head':
        n_heads = 1

    model = DecoderOnlyLanguageModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=n_heads, d_ff=D_FF, dropout=DROPOUT
    )

    if args.experiment == 'no_pe':
        model.pos_encoding = nn.Identity()
        print("实验修改: 移除了 Positional Encoding")

    if args.experiment == 'no_res':
        # (确保 n_heads 在这里是正确的)
        new_decoder_layers = nn.ModuleList([
            NoResidualCausalDecoderBlock(D_MODEL, n_heads, D_FF, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        model.layers = new_decoder_layers
        print("实验修改: 移除了 Residual Connections")

    model.to(device)

    # 5. 设置优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-9,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. 运行实验
    run_experiment(
        model, optimizer, scheduler, device, args.epochs,
        train_data, val_data, args.bptt, VOCAB_SIZE,
        experiment_name=args.experiment
    )


if __name__ == '__main__':
    main()