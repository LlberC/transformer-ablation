import json
import matplotlib.pyplot as plt
import os


def plot_results():
    results_dir = 'results'
    experiment_files = [
        'baseline_metrics.json',
        'no_pe_metrics.json',
        'no_res_metrics.json',
        'single_head_metrics.json'
    ]

    labels = {
        'baseline': 'Baseline (Decoder-Only LM)',
        'no_pe': 'Ablation: No Positional Encoding',
        'no_res': 'Ablation: No Residuals',
        'single_head': 'Ablation: Single Head (h=1)'
    }

    plt.figure(figsize=(10, 6))

    num_epochs = 0

    for filename in experiment_files:
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"警告: 找不到指标文件 {filepath}。跳过此实验。")
            continue

        with open(filepath, 'r') as f:
            metrics = json.load(f)

        val_losses = metrics['val_losses']
        exp_name_key = metrics['experiment_name']
        label = labels.get(exp_name_key, exp_name_key)

        plt.plot(range(1, len(val_losses) + 1), val_losses, label=label)
        num_epochs = max(num_epochs, len(val_losses))

    if num_epochs > 0:
        plt.title('Transformer Ablation Study: Validation Loss (Tiny Shakespeare)')
        plt.xlabel('Epoch')
        plt.ylabel('Average Validation Cross Entropy Loss')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(results_dir, 'ablation_study_loss_curve.png')
        plt.savefig(save_path)
        print(f"汇总图表已保存至: {save_path}")
        # plt.show() # 在 CI/服务器上运行时注释掉
    else:
        print("错误: 没有可绘制的数据。")


if __name__ == '__main__':
    plot_results()