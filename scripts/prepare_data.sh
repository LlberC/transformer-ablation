#!/bin/bash
# 这个脚本用于下载 Tiny Shakespeare 数据集

# 检查 input.txt 是否已存在
if [ -f "input.txt" ]; then
    echo "input.txt 已存在, 跳过下载。"
else
    echo "正在下载 input.txt..."
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    if [ $? -eq 0 ]; then
        echo "下载成功。"
    else
        echo "下载失败。请检查 wget 或网络连接。"
        exit 1
    fi
fi