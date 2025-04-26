import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_aspect_metrics_comparison(metrics_history, epoch_num):
    # 验证输入参数
    if not isinstance(metrics_history, (list, tuple)):
        print(f"Error: metrics_history must be a list or tuple, got {type(metrics_history)}")
        return
    
    if not metrics_history:
        print("Error: metrics_history is empty")
        return
    
    if not isinstance(epoch_num, (int, np.integer)):
        print(f"Error: epoch_num must be an integer, got {type(epoch_num)}")
        return
    
    # 获取所有方面名称
    aspects = [
        'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
        'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
        'Price#Level', 'Price#Cost_effective', 'Price#Discount',
        'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
        'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
    ]
    
    # 提取当前epoch的指标
    if epoch_num < 0:
        print(f"Error: epoch_num cannot be negative, got {epoch_num}")
        return
    
    if epoch_num >= len(metrics_history):
        print(f"Error: epoch {epoch_num} is out of range. metrics_history length: {len(metrics_history)}")
        return
    
    metrics = metrics_history[epoch_num]
    if not isinstance(metrics, dict):
        print(f"Error: metrics for epoch {epoch_num} is not a dictionary. Type: {type(metrics)}")
        return
    
    # 创建性能指标矩阵
    metrics_matrix = np.zeros((len(aspects), 5))
    missing_aspects = []
    
    for i, aspect in enumerate(aspects):
        if aspect not in metrics:
            missing_aspects.append(aspect)
            continue
            
        if not isinstance(metrics[aspect], dict):
            print(f"Warning: Invalid metrics format for aspect '{aspect}' in epoch {epoch_num}")
            continue
            
        try:
            metrics_matrix[i] = [
                float(metrics[aspect].get('accuracy', 0)),
                float(metrics[aspect].get('precision', 0)),
                float(metrics[aspect].get('recall', 0)),
                float(metrics[aspect].get('specificity', 0)),
                float(metrics[aspect].get('f1', 0))
            ]
        except (ValueError, TypeError) as e:
            print(f"Warning: Error converting metrics to float for aspect '{aspect}': {e}")
            continue
    
    if missing_aspects:
        print(f"Warning: Missing metrics for aspects: {', '.join(missing_aspects)}")
    
    # 创建热力图
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        metrics_matrix,
        xticklabels=['ACC', 'PPV', 'TPR', 'TNR', 'F1'],
        yticklabels=aspects,
        annot=True,
        fmt='.3f',
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        center=0.5
    )
    plt.xlabel('评估指标')
    plt.ylabel('评估维度')
    
    plt.title(f'第{epoch_num}轮训练 - 各维度评估指标对比分析', fontsize=14, pad=20)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = 'try5_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存图表
    output_path = os.path.join(output_dir, f'aspect_metrics_comparison_epoch_{epoch_num}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_aspect_metrics_comparison(metrics_by_class, epoch_name):
    """生成每个epoch的性能指标热力图"""
    # 准备数据
    aspects = list(metrics_by_class.keys())
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    data = np.zeros((len(aspects), len(metrics)))
    
    for i, aspect in enumerate(aspects):
        for j, metric in enumerate(metrics):
            data[i, j] = metrics_by_class[aspect][metric]
    
    # 创建热力图
    plt.figure(figsize=(12, 16))
    sns.heatmap(data, 
                xticklabels=metrics,
                yticklabels=aspects,
                annot=True,  # 显示数值
                fmt='.3f',   # 数值格式
                cmap='YlOrRd',  # 使用红色系配色
                vmin=0, 
                vmax=1)
    
    plt.title(f'Aspect-wise Metrics Comparison - {epoch_name}')
    plt.xlabel('Metrics')
    plt.ylabel('Aspects')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join('try5_output', f'aspect_metrics_comparison_{epoch_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

def plot_loss_curves(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('try5_output/loss_curves.png')
    plt.close()

def plot_metrics_trends(metrics_history):
    """绘制训练过程中的指标变化趋势"""
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = []
        for epoch_metrics in metrics_history:
            avg_value = np.mean([aspect_metrics[metric] 
                               for aspect_metrics in epoch_metrics.values()])
            values.append(avg_value)
        
        plt.plot(range(1, len(values) + 1), values)
        plt.title(f'Average {metric.capitalize()} Over Training')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.savefig(f'try5_output/{metric}_trend.png')
        plt.close()