import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, f1_scores):
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
    plt.savefig('training_curves.png')
    plt.close()