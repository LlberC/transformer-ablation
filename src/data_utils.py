import torch


def batchify(data, bsz, device):
    """
    将一维的（1D）数据重塑为 (seq_len, bsz) 的形状。
    """
    seq_len = data.size(0) // bsz
    data = data.narrow(0, 0, seq_len * bsz)
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source, i, bptt):
    """
    从 source 数据中获取一个 (data, target) 批次。
    target 是 data 向右偏移一位。
    """
    seq_len = min(bptt, source.size(0) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    # (Seq, Batch) -> (Batch, Seq)
    return data.t().contiguous(), target.t().contiguous()


def load_data(device, batch_size, eval_batch_size, file_path='input.txt'):
    """
    加载、处理和批处理 Tiny Shakespeare 数据集。

    返回:
        train_data (Tensor): 批处理后的训练数据 (Seq_len, Batch)
        val_data (Tensor): 批处理后的验证数据 (Seq_len, Batch)
        vocab_size (int): 词汇表大小
    """

    # 1. 加载原始文本数据
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"成功加载 '{file_path}'")
    except FileNotFoundError:
        print(f"错误: '{file_path}' 未找到。")
        print("请先运行 'bash scripts/prepare_data.sh' 下载数据。")
        # 抛出异常而不是退出，以便调用者可以处理
        raise
    except Exception as e:
        print(f"加载数据时发生未知错误: {e}")
        raise

    # 2. 创建词汇表
    chars = sorted(list(set(text)))
    VOCAB_SIZE = len(chars)
    print(f"语料库共有 {len(text)} 个字符，唯一字符 {VOCAB_SIZE} 个。")

    # 3. 创建字符到索引的映射
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    # (decode 未在训练中使用，但保留在这里以便理解)
    # decode = lambda l: ''.join([itos[i] for i in l])

    # 4. 将整个数据集编码为张量，并进行训练/验证集划分
    full_data = torch.tensor(encode(text), dtype=torch.long)
    n = len(full_data)
    train_data = full_data[:int(n * 0.9)]
    val_data = full_data[int(n * 0.9):]

    # 5. 使用传入的参数进行批处理
    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)

    print(f"Train data shape: {train_data.shape} (Seq, Batch)")
    print(f"Val data shape: {val_data.shape} (Seq, Batch)")

    return train_data, val_data, VOCAB_SIZE