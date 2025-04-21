import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import jieba
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix
from collections import Counter
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from plot_training_curves import plot_training_curves

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')

# 定义情感标签
aspect_categories = [
    'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
    'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
    'Price#Level', 'Price#Cost_effective', 'Price#Discount',
    'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
    'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
]

class RestaurantDataset(Dataset):
    def __init__(self, data, vocab, max_len=128):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['review'])
        words = jieba.lcut(text)
        
        # 文本转换为索引
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_len - len(tokens))
        
        # 准备标签，将原始情感标签[-2, -1, 0, 1]映射到适当的值
        labels = []
        for aspect in aspect_categories:
            original_label = float(self.data.iloc[idx][aspect])
            # 将原始标签映射到合适的值
            if original_label == -2:  # 没有提到
                label = 0.0  # 可以用0表示没有提到
            elif original_label == -1:  # 负面
                label = 0.25  # 负面情感映射到0.25
            elif original_label == 0:   # 中性
                label = 0.5   # 中性情感映射到0.5
            elif original_label == 1:   # 正面
                label = 1.0   # 正面情感映射到1.0
            else:
                # 处理异常情况
                label = 0.5   # 默认为中性
            labels.append(label)
        labels = torch.tensor(labels)
        
        return torch.tensor(tokens), labels

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attention_weights

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)
        context, attention_weights = self.attention(lstm_output)
        context = self.dropout(context)
        out = self.fc(context)
        return torch.sigmoid(out), attention_weights

def calculate_metrics(y_true, y_pred):
    """计算各种性能指标"""
    # 将连续值转换为类别
    # 首先将预测值和真实值转换回原始标签类别
    y_true_classes = []
    y_pred_classes = []
    
    for true_val, pred_val in zip(y_true, y_pred):
        # 将真实标签转换为类别
        if abs(true_val - 0.0) < 1e-5:
            true_class = -2  # 没有提到
        elif abs(true_val - 0.25) < 1e-5:
            true_class = -1  # 负面
        elif abs(true_val - 0.5) < 1e-5:
            true_class = 0   # 中性
        elif abs(true_val - 1.0) < 1e-5:
            true_class = 1   # 正面
        else:
            # 异常情况，可能是由于舍入误差
            true_class = 0   # 默认为中性
        
        # 将预测标签转换为类别
        if pred_val < 0.125:
            pred_class = -2  # 没有提到
        elif pred_val < 0.375:
            pred_class = -1  # 负面
        elif pred_val < 0.75:
            pred_class = 0   # 中性
        else:
            pred_class = 1   # 正面
        
        y_true_classes.append(true_class)
        y_pred_classes.append(pred_class)
    
    # 计算准确率 - 完全匹配的比例
    accuracy = np.mean(np.array(y_true_classes) == np.array(y_pred_classes))
    
    # 对于每个类别分别计算指标
    # 我们将每个类别视为一个二分类问题进行评估
    metrics = {}
    
    for class_label, class_name in [(-2, "没有提到"), (-1, "负面"), (0, "中性"), (1, "正面")]:
        # 创建二分类任务的标签
        binary_true = [1 if y == class_label else 0 for y in y_true_classes]
        binary_pred = [1 if y == class_label else 0 for y in y_pred_classes]
        
        # 如果该类别没有样本，跳过计算
        if sum(binary_true) == 0 and sum(binary_pred) == 0:
            continue
            
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()
        
        # 计算各项指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1
        }
    
    # 计算总体指标
    avg_metrics = {
        'accuracy': accuracy,
        'precision': np.mean([m['precision'] for m in metrics.values()]),
        'recall': np.mean([m['recall'] for m in metrics.values()]),
        'specificity': np.mean([m['specificity'] for m in metrics.values()]),
        'f1': np.mean([m['f1'] for m in metrics.values()])
    }
    
    return avg_metrics

def evaluate(model, data_loader, criterion, device):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 将输出作为连续值保存，供calculate_metrics处理
            predictions.extend(output.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算整体性能指标
    avg_metrics = calculate_metrics(targets.flatten(), predictions.flatten())
    
    # 打印性能指标
    print("\n整体性能指标：")
    print(f"准确率 (ACC): {avg_metrics['accuracy']:.4f}")
    print(f"平均精确率 (PPV): {avg_metrics['precision']:.4f}")
    print(f"平均灵敏度 (TPR): {avg_metrics['recall']:.4f}")
    print(f"平均特异度 (TNR): {avg_metrics['specificity']:.4f}")
    print(f"平均F1分数: {avg_metrics['f1']:.4f}")
    
    return avg_metrics, total_loss / len(data_loader)

def plot_metrics(metrics_history):
    """绘制性能指标变化曲线"""
    plt.figure(figsize=(15, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot([m[metric] for m in metrics_history], label=metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} over epochs')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('metrics_curves.png')
    plt.close()

def main():
    # 加载数据
    train_df = pd.read_csv('restaurant_comment_data/train.csv')
    test_df = pd.read_csv('restaurant_comment_data/test.csv')
    dev_df = pd.read_csv('restaurant_comment_data/dev.csv')
    
    # 构建词汇表
    vocab = build_vocab(train_df['review'])
    
    # 创建数据集
    train_dataset = RestaurantDataset(train_df, vocab)
    dev_dataset = RestaurantDataset(dev_df, vocab)
    test_dataset = RestaurantDataset(test_df, vocab)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = BiLSTMModel(
        vocab_size=len(vocab),
        embed_dim=300,
        hidden_dim=128,
        num_classes=len(aspect_categories)
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # 训练模型
    num_epochs = 10
    best_val_f1 = 0
    train_losses = []
    val_losses = []
    metrics_history = []
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_loss = evaluate(model, dev_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_history.append(val_metrics)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 绘制训练曲线和性能指标曲线
    plot_training_curves(train_losses, val_losses, [m['f1'] for m in metrics_history])
    plot_metrics(metrics_history)
    
    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics, test_loss = evaluate(model, test_loader, criterion, device)
    print(f'\n测试集性能：')
    print(f'Test Loss: {test_loss:.4f}')

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for data, target in tqdm(data_loader, desc='Training'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def build_vocab(texts, min_freq=2):
    """构建词汇表
    Args:
        texts: 文本数据Series
        min_freq: 最小词频阈值
    Returns:
        vocab: 词到索引的映射字典
    """
    # 特殊标记
    special_tokens = ['<PAD>', '<UNK>']
    
    # 分词并统计词频
    word_counts = Counter()
    for text in texts:
        words = jieba.lcut(str(text))
        word_counts.update(words)
    
    # 过滤低频词
    valid_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # 构建词汇表
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    for i, word in enumerate(valid_words, len(special_tokens)):
        vocab[word] = i
    
    return vocab

if __name__ == '__main__':
    main()