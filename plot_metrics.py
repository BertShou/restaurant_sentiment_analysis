import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_aspect_metrics_by_epoch(metrics_by_class, epoch, output_dir):
    # 准备数据
    aspects = list(metrics_by_class.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    metrics_labels = ['ACC', 'PPV', 'TPR', 'TNR', 'F1']
    
    # 创建数据矩阵
    data_matrix = np.zeros((len(aspects), len(metrics_names)))
    for i, aspect in enumerate(aspects):
        for j, metric in enumerate(metrics_names):
            data_matrix[i, j] = metrics_by_class[aspect][metric]
    
    # 设置图形样式和大小
    plt.style.use('default')
    plt.figure(figsize=(15, 12))
    
    # 创建热力图
    sns.heatmap(data_matrix,
                xticklabels=metrics_labels,
                yticklabels=aspects,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Score'})
    
    # 设置标题和标签
    plt.title(f'Performance Metrics by Aspect Category (Epoch {epoch})', pad=20)
    plt.xlabel('Metrics')
    plt.ylabel('Aspect Categories')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(output_dir, f'aspect_metrics_comparison_epoch_{epoch}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves_by_epoch(train_losses, val_losses, f1_scores, output_dir):
    """绘制训练曲线，包括训练损失、验证损失和F1分数"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制F1分数曲线
    plt.subplot(1, 2, 2)
    plt.plot(f1_scores, label='F1 Score', color='green')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path)
    plt.close()