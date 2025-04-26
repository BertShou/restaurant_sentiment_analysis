import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_aspect_metrics(metrics_by_class, output_path):
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
    plt.title('Performance Metrics by Aspect Category', pad=20)
    plt.xlabel('Metrics')
    plt.ylabel('Aspect Categories')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()