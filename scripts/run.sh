#!/bin/bash

# 确保脚本在出错时停止
set -e

# 设置可复现的随机种子
SEED=42
EPOCHS=10
LR=3e-4

# 0. 准备数据
echo "--- 准备数据 ---"
bash scripts/prepare_data.sh

# 1. 运行基准实验
echo "\n--- 运行基准 (Baseline) 实验 ---"
python src/main.py --experiment baseline --seed $SEED --epochs $EPOCHS --lr $LR

# 2. 运行消融 1: 无位置编码
echo "\n--- 运行消融: 无位置编码 (No PE) ---"
python src/main.py --experiment no_pe --seed $SEED --epochs $EPOCHS --lr $LR

# 3. 运行消融 2: 无残差连接
echo "\n--- 运行消融: 无残差连接 (No Residuals) ---"
python src/main.py --experiment no_res --seed $SEED --epochs $EPOCHS --lr $LR

# 4. 运行消融 3: 单头注意力
echo "\n--- 运行消融: 单头注意力 (Single Head) ---"
python src/main.py --experiment single_head --seed $SEED --epochs $EPOCHS --lr $LR

# 5. 汇总所有结果并绘图
echo "\n--- 汇总结果并生成图表 ---"
python src/plot_all.py

echo "\n--- 所有实验已完成! ---"
echo "最终图表已保存至: results/ablation_study_loss_curve.png"