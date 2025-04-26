import os
import random  # 添加 random 模块导入
import warnings
from collections import Counter

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from plot_training_curves import plot_training_curves
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
        self.indices = list(range(len(data)))  # 添加索引列表

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

    def shuffle_data(self):
        # 打乱索引顺序
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        # 使用打乱后的索引访问数据
        shuffled_idx = self.indices[idx]
        text = str(self.data.iloc[shuffled_idx]['review'])

        # 标签也需要使用打乱后的索引
        labels = []
        for aspect in aspect_categories:
            label = float(self.data.iloc[shuffled_idx][aspect])  # 修改这里
            label = max(0.0, min(1.0, label))
            labels.append(label)
        words = jieba.lcut(text)

        # 文本转换为索引
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_len - len(tokens))

        # 准备标签，确保标签值在[0,1]范围内
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_output, _ = self.lstm(embedded)
        context, attention_weights = self.attention(lstm_output)
        context = self.dropout(context)
        out = self.fc(context)
        return self.sigmoid(out), attention_weights


def calculate_metrics(y_true, y_pred):
    """计算各种性能指标"""
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 计算各项指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1
    }


def evaluate(model, data_loader, criterion, device, metrics_history=None, current_epoch=None):
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
            pred = (output > 0.5).float()
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # 计算每个类别的性能指标
    metrics_by_class = {}
    for i, aspect in enumerate(aspect_categories):
        metrics = calculate_metrics(targets[:, i], predictions[:, i])
        metrics_by_class[aspect] = metrics

    # 计算平均性能指标
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in metrics_by_class.values()]),
        'precision': np.mean([m['precision'] for m in metrics_by_class.values()]),
        'recall': np.mean([m['recall'] for m in metrics_by_class.values()]),
        'specificity': np.mean([m['specificity'] for m in metrics_by_class.values()]),
        'f1': np.mean([m['f1'] for m in metrics_by_class.values()])
    }

    # 打印每个类别的详细性能指标
    print("\n每个类别的性能指标：")
    for aspect, metrics in metrics_by_class.items():
        print(f"\n{aspect}:")
        print(f"  准确率 (ACC): {metrics['accuracy']:.4f}")
        print(f"  精确率 (PPV): {metrics['precision']:.4f}")
        print(f"  灵敏度 (TPR): {metrics['recall']:.4f}")
        print(f"  特异度 (TNR): {metrics['specificity']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")

    # 打印平均性能指标
    print("\n平均性能指标：")
    print(f"平均准确率 (ACC): {avg_metrics['accuracy']:.4f}")
    print(f"平均精确率 (PPV): {avg_metrics['precision']:.4f}")
    print(f"平均灵敏度 (TPR): {avg_metrics['recall']:.4f}")
    print(f"平均特异度 (TNR): {avg_metrics['specificity']:.4f}")
    print(f"平均F1分数: {avg_metrics['f1']:.4f}")

    # 生成当前epoch的热力图
    if current_epoch is not None:
        from plot_epoch_metrics import plot_aspect_metrics_comparison
        os.makedirs('analysis_output', exist_ok=True)  # 确保输出目录存在
        plot_aspect_metrics_comparison(
            metrics_by_class,
            str(current_epoch + 1)  # 修改为字符串格式
        )

    if metrics_history is not None:
        metrics_history.append(metrics_by_class)

    return avg_metrics, total_loss / len(data_loader)


# 1. 添加数据增强相关代码
def augment_text(text):
    """文本数据增强"""
    augmented_texts = []
    words = jieba.lcut(str(text))

    # 随机删除词
    if len(words) > 3:
        delete_pos = random.randint(0, len(words) - 1)
        aug_text = ''.join(words[:delete_pos] + words[delete_pos + 1:])
        augmented_texts.append(aug_text)

    # 随机替换同义词
    synonyms = {
        '好吃': ['美味', '可口', '美味可口'],
        '服务': ['服务态度', '服务质量'],
        '环境': ['氛围', '场所', '装修'],
        # 可以添加更多同义词
    }
    for i, word in enumerate(words):
        if word in synonyms:
            aug_words = words.copy()
            aug_words[i] = random.choice(synonyms[word])
            augmented_texts.append(''.join(aug_words))

    return augmented_texts


# 1. 优化模型结构
class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(EnhancedBiLSTMModel, self).__init__()

        # 使用预训练词向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)

        # 双向LSTM层
        self.lstm_layers = 2
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate if self.lstm_layers > 1 else 0
        )

        # 多头注意力机制
        self.num_heads = 4
        self.attention_heads = nn.ModuleList([
            Attention(hidden_dim) for _ in range(self.num_heads)
        ])

        # 残差连接和层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 * self.num_heads)
        self.dropout = nn.Dropout(dropout_rate)

        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * self.num_heads, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        lstm_output, _ = self.lstm(embedded)

        # 多头注意力处理
        attention_outputs = []
        attention_weights_list = []
        for attention_head in self.attention_heads:
            context, attention_weights = attention_head(lstm_output)
            attention_outputs.append(context)
            attention_weights_list.append(attention_weights)

        # 合并多头注意力的输出
        combined_context = torch.cat(attention_outputs, dim=1)

        # 残差连接和层归一化
        normalized_context = self.layer_norm(combined_context)
        dropped_context = self.dropout(normalized_context)

        # 分类
        output = self.classifier(dropped_context)

        return output, attention_weights_list


def main():
    # 加载数据
    train_df = pd.read_csv('restaurant_comment_data/train.csv')
    test_df = pd.read_csv('restaurant_comment_data/test.csv')
    dev_df = pd.read_csv('restaurant_comment_data/dev.csv')

    # 设置设备（移到最前面）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU 进行计算")

    # 数据增强
    augmented_reviews = []
    augmented_labels = []
    for idx, row in tqdm(train_df.iterrows(), desc='数据增强'):
        aug_texts = augment_text(row['review'])
        for aug_text in aug_texts:
            augmented_reviews.append(aug_text)
            # TODO 这里要检查一下，情感表情有4个，不应该只是在0和1之间？
            # 确保标签值在[0,1]范围内
            labels = [max(0.0, min(1.0, float(row[aspect]))) for aspect in aspect_categories]
            augmented_labels.append(labels)

    # 合并原始数据和增强数据
    aug_df = pd.DataFrame({
        'review': augmented_reviews,
        **{aspect: [labels[i] for labels in augmented_labels]
           for i, aspect in enumerate(aspect_categories)}
    })
    train_df = pd.concat([train_df, aug_df], ignore_index=True)

    # 构建词汇表
    vocab = build_vocab(train_df['review'])

    # 使用增强后的模型（删除重复的模型初始化）
    model = EnhancedBiLSTMModel(
        vocab_size=len(vocab),
        embed_dim=300,
        hidden_dim=128,
        num_classes=len(aspect_categories),
        dropout_rate=0.3
    ).to(device)

    # 使用学习率调度器
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2, verbose=True
    )

    # 创建数据集
    train_dataset = RestaurantDataset(train_df, vocab)
    dev_dataset = RestaurantDataset(dev_df, vocab)
    test_dataset = RestaurantDataset(test_df, vocab)

    # 创建数据加载器
    # 移除原有的设备设置代码
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载器添加 pin_memory
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              pin_memory=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=32,
                            pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             pin_memory=True, num_workers=4)

    # 训练参数设置
    num_epochs = 10  # 添加这行
    best_val_f1 = 0
    train_losses = []
    val_losses = []
    metrics_history = []

    # 修改优化器参数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # 降低学习率
        weight_decay=0.1  # 增加权重衰减
    )

    # 添加早停机制
    early_stopping_patience = 3
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, val_loss = evaluate(model, dev_loader, criterion, device,
                                         metrics_history, current_epoch=epoch)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            no_improve_count += 1

        if no_improve_count >= early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

    # 删除这个循环，因为已经在每个epoch中生成了热力图
    # for epoch in range(len(metrics_history)):
    #     if metrics_history[epoch]:
    #         plot_aspect_metrics_comparison(metrics_history[epoch], epoch)

    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        torch.save(model.state_dict(), 'best_model.pth')

    # 绘制训练曲线和性能指标曲线
    from plot_epoch_metrics import plot_aspect_metrics_comparison, plot_loss_curves, plot_metrics_trends

    # 修改循环，只为有效的metrics生成热力图
    for epoch in range(len(metrics_history)):
        if metrics_history[epoch]:  # 只有当metrics不为空时才生成图
            plot_aspect_metrics_comparison(metrics_history[epoch], epoch)

    # 确保输出目录存在
    os.makedirs('analysis_output', exist_ok=True)

    # 绘制损失曲线
    plot_loss_curves(train_losses, val_losses)

    # 绘制训练过程中的指标变化趋势
    plot_metrics_trends(metrics_history)

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics, test_loss = evaluate(model, test_loader, criterion, device)
    print(f'\n测试集性能：')
    print(f'Test Loss: {test_loss:.4f}')

    # 添加模型性能评价总结
    print("\n=== 模型性能评价总结 ===")
    print("\n1. 整体表现：")
    performance_level = "优秀" if test_metrics['f1'] > 0.8 else "良好" if test_metrics['f1'] > 0.6 else "一般"
    print(f"模型整体表现{performance_level}，平均F1分数为{test_metrics['f1']:.4f}")

    print("\n2. 各项指标分析：")
    print(f"- 准确率：{test_metrics['accuracy']:.4f} (反映模型的整体预测准确性)")
    print(f"- 精确率：{test_metrics['precision']:.4f} (反映模型预测为正例的可信度)")
    print(f"- 召回率：{test_metrics['recall']:.4f} (反映模型捕获正例的能力)")
    print(f"- 特异度：{test_metrics['specificity']:.4f} (反映模型识别负例的能力)")

    print("\n3. 模型特点：")
    if test_metrics['precision'] > test_metrics['recall']:
        print("- 模型倾向于保守预测，假阳性率较低")
    else:
        print("- 模型倾向于激进预测，召回率较高")

    print("\n4. 改进建议：")
    if test_metrics['f1'] < 0.6:
        print("- 考虑增加训练数据或优化模型结构")
        print("- 可能需要调整类别权重以处理数据不平衡")
    elif test_metrics['precision'] < 0.6:
        print("- 建议提高模型的精确率，减少假阳性预测")
    elif test_metrics['recall'] < 0.6:
        print("- 建议提高模型的召回率，减少漏判")
    else:
        print("- 模型表现良好，可以考虑进一步微调以提升特定方面的性能")


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    # 添加梯度裁剪参数定义
    max_grad_norm = 1.0  # 在函数开始处定义

    # 重新打乱数据加载器
    data_loader.dataset.shuffle_data()

    # 使用 CUDA 事件来同步和计时
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    for batch in tqdm(data_loader, desc='Training'):
        texts, labels = batch
        # 非阻塞地将数据转移到 GPU
        texts = texts.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs, _ = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item()

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        print(f'Batch processing time: {start_event.elapsed_time(end_event):.2f} ms')

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
    special_tokens = ['<PAD>', '<UNK>']
    main()
