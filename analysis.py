import os
import random
import warnings
from collections import Counter

import jieba  # 中文分词库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix  # 评估指标
from torch.optim.lr_scheduler import OneCycleLR  # 学习率调度器
from torch.utils.data import Dataset, DataLoader  # PyTorch数据处理工具
from tqdm import tqdm  # 进度条库

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')  # 忽略不必要的警告信息

# --- 数据和标签定义 ---
# 定义18个细粒度的方面类别
aspect_categories = [
    'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
    'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
    'Price#Level', 'Price#Cost_effective', 'Price#Discount',
    'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
    'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
]

# 定义情感标签的原始值到类别索引的映射
# 原始情感标签: 1 (正面), 0 (中性), -1 (负面), -2 (未提及)
# 映射后的类别索引: 0 (未提及), 1 (负面), 2 (中性), 3 (正面)
sentiment_mapping = {
    1: 3,  # 正面
    0: 2,  # 中性
    -1: 1,  # 负面
    -2: 0  # 未提及
}
sentiment_names = ["未提及", "负面", "中性", "正面"]  # 情感类别名称，用于结果展示
num_sentiment_classes = len(sentiment_names)  # 情感类别总数，应为4


# --- PyTorch Dataset 定义 ---
class RestaurantDataset(Dataset):
    """自定义餐厅评论数据集类"""

    def __init__(self, data, vocab, max_len=128):
        """
        Args:
            data (pd.DataFrame): 包含评论和标签的DataFrame
            vocab (dict): 词汇表，词到索引的映射
            max_len (int): 文本序列的最大长度
        """
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.indices = list(range(len(data)))  # 创建索引列表，用于后续打乱数据

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.data)

    def shuffle_data(self):
        """打乱数据集的索引顺序，用于训练时的数据增强"""
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        """根据索引获取单个样本数据"""
        # 使用打乱后的索引访问数据，确保每次epoch的顺序不同
        shuffled_idx = self.indices[idx]
        text = str(self.data.iloc[shuffled_idx]['review'])  # 获取评论文本

        # 处理标签：读取每个方面的原始情感标签，并映射到对应的类别索引
        labels = []
        for aspect in aspect_categories:
            try:
                raw_label = int(self.data.iloc[shuffled_idx][aspect])  # 原始标签值
            except ValueError:
                # 处理标签转换失败的情况 (例如NaN或非数字)，默认为"未提及"
                raw_label = -2

                # 使用预定义的sentiment_mapping进行映射，找不到则默认为"未提及"对应的索引
            class_index = sentiment_mapping.get(raw_label, sentiment_mapping[-2])
            labels.append(class_index)

        # 使用jieba进行中文分词
        words = jieba.lcut(text)

        # 将分词后的文本转换为词汇表中的索引序列
        tokens = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]  # <UNK>表示未登录词
        # 对序列进行填充或截断，使其达到max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_len - len(tokens))  # <PAD>表示填充符

        # 将标签列表和文本索引序列转换为PyTorch张量
        # 标签使用LongTensor，因为CrossEntropyLoss期望的是类别索引
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)  # token也用long，因为是索引

        return tokens_tensor, labels_tensor


# --- 模型定义 (注意力机制和主模型) ---
class Attention(nn.Module):
    """Luong风格的注意力机制实现"""

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 注意力计算层：线性变换 -> Tanh激活 -> 再线性变换得到注意力分数
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # BiLSTM输出是hidden_dim*2
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # 层归一化，稳定训练

    def forward(self, lstm_output):
        """
        Args:
            lstm_output (torch.Tensor): LSTM的输出, shape (batch_size, seq_len, hidden_dim * 2)
        Returns:
            context (torch.Tensor): 加权后的上下文向量, shape (batch_size, hidden_dim * 2)
            attention_weights (torch.Tensor): 注意力权重, shape (batch_size, seq_len)
        """
        normalized_output = self.layer_norm(lstm_output)  # 先进行层归一化
        # 计算注意力权重，squeeze(-1)移除最后一个维度1
        attention_weights = self.attention(normalized_output).squeeze(-1)
        # 对权重进行softmax归一化，使其和为1
        attention_weights = torch.softmax(attention_weights, dim=1)
        # 计算上下文向量：将权重与LSTM输出加权求和
        # bmm进行批处理矩阵乘法: (batch, 1, seq_len) * (batch, seq_len, hidden_dim*2) -> (batch, 1, hidden_dim*2)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attention_weights


class EnhancedBiLSTMModel(nn.Module):
    """增强型双向LSTM模型，结合多头注意力和门控机制"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_aspects, num_sentiment_classes, dropout_rate=0.3):
        super(EnhancedBiLSTMModel, self).__init__()
        self.num_aspects = num_aspects
        self.num_sentiment_classes = num_sentiment_classes
        self.hidden_dim = hidden_dim # 保存 hidden_dim 为实例属性

        # 词嵌入层，padding_idx=0表示索引0的词（<PAD>）不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)  # Embedding层后的dropout

        # 双向LSTM层，增加层数和隐藏维度以增强模型表达能力
        self.lstm_layers = 2
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=self.lstm_layers,
            bidirectional=True,  # 双向LSTM
            batch_first=True,  # 输入输出张量的第一个维度是batch_size
            dropout=dropout_rate if self.lstm_layers > 1 else 0  # 层间的dropout
        )

        # 多头注意力机制，每个头关注文本的不同部分
        self.num_heads = 8
        self.attention_heads = nn.ModuleList([
            Attention(hidden_dim) for _ in range(self.num_heads)
        ])

        # 层归一化和Dropout层
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)  # LSTM输出后
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2 * self.num_heads)  # 多头注意力拼接后
        self.dropout = nn.Dropout(dropout_rate)

        # 特征融合层，使用门控机制动态融合多头注意力的输出
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 * self.num_heads, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),  # GELU激活函数，表现通常优于ReLU
            nn.Dropout(dropout_rate)
        )
        # 门控机制，学习哪些融合特征更重要
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 * self.num_heads, hidden_dim * 4),
            nn.Sigmoid()
        )

        # 多层感知机(MLP)分类器，进一步提取和转换特征
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # 输出层，为每个方面预测其4种情感类别的分数（logits）
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),  # 输出层使用较低的dropout
            # 输出特征数为：方面数 * 每个方面的情感类别数
            nn.Linear(hidden_dim // 2, self.num_aspects * self.num_sentiment_classes)
            # CrossEntropyLoss期望原始logits，所以这里不需要Sigmoid或Softmax
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 1. 词嵌入
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)  # (batch_size, seq_len, embed_dim)

        # 2. LSTM处理
        lstm_output, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        lstm_output = self.layer_norm1(lstm_output)  # 层归一化

        # 3. 多头注意力处理
        attention_outputs = []
        attention_weights_list = []  # 用于可能的注意力可视化或分析
        for attention_head in self.attention_heads:
            context, attention_weights = attention_head(lstm_output)
            attention_outputs.append(context)
            attention_weights_list.append(attention_weights)

        # 合并多头注意力的输出 (batch_size, hidden_dim * 2 * num_heads)
        combined_context = torch.cat(attention_outputs, dim=1)
        # 注意：这里原文有一个 lstm_output.repeat 的操作，但在当前上下文中，
        # combined_context 应该是直接从多头注意力得到的，其维度已经是融合了多头信息的。
        # 如果需要残差连接，应确保维度匹配或使用合适的投影。
        # 当前的 combined_context 维度是 (batch_size, hidden_dim * 2 * num_heads) (因为每个context是 hidden_dim*2, cat了num_heads个)
        # 但原始论文或常见做法可能是每个头的输出是 hidden_dim, 然后拼接成 hidden_dim * num_heads, 或者平均/加权求和。
        # 这里的 Attention 输出是 hidden_dim * 2, 拼接后是 hidden_dim * 2 * num_heads。
        # 因此 layer_norm2 的输入维度是正确的。
        normalized_context = self.layer_norm2(combined_context)
        dropped_context = self.dropout(normalized_context)

        # 4. 使用门控机制进行特征融合
        gate_values = self.gate(dropped_context)  # (batch_size, hidden_dim * 4)
        fused_features = self.feature_fusion(dropped_context) * gate_values  # (batch_size, hidden_dim * 4)

        # 5. 多层感知机处理
        mlp_output = self.mlp(fused_features)  # (batch_size, hidden_dim)

        # 6. 残差连接和最终输出
        # 原文的残差连接 fused_features[:, :mlp_output.size(1)] 暗示 fused_features 可能维度更大
        # 但这里 fused_features 和 mlp_output 的输入/输出维度需要匹配才能简单相加。
        # 当前 fused_features 是 (hidden_dim * 4), mlp_output 是 (hidden_dim)
        # 若要残差连接，需要调整 MLP 结构或 fused_features 维度，或使用投影。
        # 假设这里的意图是直接将 MLP 输出送入最终的 output 层
        # 或者是 self.mlp 的最后输出维度应与 self.output 的第一个线性层输入匹配。
        # 当前 self.mlp 输出 hidden_dim, self.output 第一个线性层输入也是 hidden_dim。这是匹配的。
        # 残差连接 mlp_output + fused_features[:, :mlp_output.size(1)] 的意图可能是想让原始的 fused_features 的一部分信息也传过去。
        # 如果 fused_features 是 (batch, D1) 而 mlp_output 是 (batch, D2) 且 D2 < D1，那么这个切片才有意义。
        # 当前 fused_features 是 (batch, hidden_dim*4), mlp_output 是 (batch, hidden_dim)。
        # 所以 mlp_output + fused_features[:, :hidden_dim] 是可行的。
        output_input = mlp_output + fused_features[:, :self.hidden_dim]  # 构造输出层的输入
        model_output_flat = self.output(output_input)  # (batch_size, num_aspects * num_sentiment_classes)

        # 7. Reshape 输出以匹配 (batch_size, num_aspects, num_sentiment_classes)
        final_output_reshaped = model_output_flat.view(batch_size, self.num_aspects, self.num_sentiment_classes)

        return final_output_reshaped, attention_weights_list


# --- 辅助函数 (构建词汇表、文本增强、评估指标计算) ---

def build_vocab(texts, min_freq=2):
    """构建词汇表"""
    special_tokens = ['<PAD>', '<UNK>']  # 定义特殊标记：填充符和未知词
    word_counts = Counter()  # 使用Counter统计词频
    for text in texts:
        words = jieba.lcut(str(text))  # 分词
        word_counts.update(words)
    # 过滤低频词，只保留出现次数大于等于min_freq的词
    valid_words = [word for word, count in word_counts.items() if count >= min_freq]
    # 构建词到索引的映射字典
    vocab = {token: i for i, token in enumerate(special_tokens)}
    for i, word in enumerate(valid_words, len(special_tokens)):
        vocab[word] = i
    return vocab


def augment_text(text):
    """简单的文本数据增强方法"""
    augmented_texts = []
    words = jieba.lcut(str(text))

    # 1. 随机删除词 (当词数大于3时)
    if len(words) > 3 and random.random() < 0.1:  # 降低触发概率
        delete_pos = random.randint(0, len(words) - 1)
        aug_text = ''.join(words[:delete_pos] + words[delete_pos + 1:])
        augmented_texts.append(aug_text)

    # 2. 随机同义词替换 (示例，实际应用需要更完善的同义词库)
    # 注意：这里的同义词库非常简单，实际效果有限
    # if len(words) > 0 and random.random() < 0.1: # 降低触发概率
    #     synonyms = {
    #         '好': ['不错', '优秀', '棒', '佳'], '差': ['糟糕', '不好', '劣'],
    #         '贵': ['昂贵', '高价'], '便宜': ['实惠', '划算']
    #         # 可以扩展更多同义词
    #     }
    #     words_copy = list(words) # 创建副本进行修改
    #     replaced = False
    #     for i, word in enumerate(words_copy):
    #         if word in synonyms and random.random() < 0.3: # 对每个可替换词再加概率
    #             words_copy[i] = random.choice(synonyms[word])
    #             replaced = True
    #     if replaced:
    #         augmented_texts.append(''.join(words_copy))

    # 3. 随机重复关键词 (示例，效果可能不佳或产生噪声)
    # if len(words) > 1 and random.random() < 0.05: # 较低概率触发
    #     sentiment_words_example = ['很', '非常', '特别', '真的'] # 示例情感加强词
    #     words_copy = list(words)
    #     inserted = False
    #     for i, word in enumerate(words_copy):
    #         if word in sentiment_words_example and random.random() < 0.3:
    #             words_copy.insert(i, word) # 重复情感词
    #             inserted = True
    #             break # 通常重复一次即可
    #     if inserted:
    #        augmented_texts.append(''.join(words_copy))

    # 注意：原始代码中 augument_text 有多个同义词替换和随机操作的片段，这里简化为一个示例。
    # 实际数据增强策略需要仔细设计和验证，避免引入过多噪声。
    # 简单的随机删除可能是一种相对安全有效的方法。

    return augmented_texts


def calculate_metrics_multiclass(y_true, y_pred, num_classes):
    """计算多分类性能指标 (针对单个方面)
    Args:
        y_true (np.array): 真实标签 (类别索引)
        y_pred (np.array): 预测标签 (类别索引)
        num_classes (int): 类别总数 (例如，情感类别数为4)
    Returns:
        dict: 包含 accuracy, precision (macro), recall (macro), f1 (macro), specificity (macro), 和 confusion_matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    # 定义用于计算指标的标签列表 (0 到 num_classes-1)
    metric_labels = list(range(num_classes))

    # 计算宏平均精确率、召回率、F1分数
    # zero_division=0 表示当分母为0时，指标记为0，避免警告或错误
    precision = precision_score(y_true, y_pred, average='macro', labels=metric_labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', labels=metric_labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', labels=metric_labels, zero_division=0)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=metric_labels)

    # 计算每个类别的特异度，然后取宏平均
    specificities = []
    if cm.shape == (num_classes, num_classes):  # 确保混淆矩阵维度正确
        for i in range(num_classes):
            tp = cm[i, i]  # 当前类的真阳性
            fp = np.sum(cm[:, i]) - tp  # 当前类的假阳性 (所有预测为该类的 - 真阳性)
            fn = np.sum(cm[i, :]) - tp  # 当前类的假阴性 (所有真实为该类的 - 真阳性)
            tn = np.sum(cm) - (tp + fp + fn)  # 当前类的真阴性 (总数 - (TP+FP+FN))

            specificity_class = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity_class)

    macro_specificity = np.mean(specificities) if specificities else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': macro_specificity,
        'confusion_matrix': cm.tolist()
    }


# --- 训练和评估函数 ---
def train_epoch(model, data_loader, criterion, optimizer, device, scheduler=None, grad_clip_value=None):
    """执行单个epoch的训练"""
    model.train()  # 设置模型为训练模式
    total_loss = 0

    # （可选）如果DataLoader没有在创建时shuffle，可以在这里shuffle Dataset的indices
    # if hasattr(data_loader.dataset, 'shuffle_data'):
    #     data_loader.dataset.shuffle_data()

    # GPU训练时，使用CUDA事件进行时间分析 (可选)
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    for batch in tqdm(data_loader, desc='训练中', leave=False):
        texts, labels = batch  # labels shape: (batch_size, num_aspects)
        # 将数据迁移到指定设备 (CPU或GPU)
        texts = texts.to(device, non_blocking=True)  # non_blocking用于CUDA数据传输优化
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()  # 清空之前的梯度
        outputs, _ = model(texts)  # 前向传播，outputs shape: (batch_size, num_aspects, num_sentiment_classes)

        # 计算损失：CrossEntropyLoss期望输入 (N, C, ...) 和目标 (N, ...)
        # C是类别数，所以需要将outputs的维度从 (batch, num_aspects, num_sent_classes) 
        # 变换为 (batch, num_sent_classes, num_aspects) 以匹配期望
        # labels的形状是 (batch, num_aspects)，正好作为目标
        loss = criterion(outputs.permute(0, 2, 1), labels)

        loss.backward()  # 反向传播计算梯度

        # 梯度裁剪，防止梯度爆炸
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()  # 更新模型参数

        # OneCycleLR等调度器需要在每个batch后更新学习率
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()  # 累加loss (loss.item()获取Python标量)

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()  # 等待所有CUDA核心完成
        print(f'Epoch训练批处理时间: {start_event.elapsed_time(end_event):.2f} ms')

    return total_loss / len(data_loader) if len(data_loader) > 0 else 0


def evaluate(model, data_loader, criterion, device, metrics_history=None, current_epoch=None):
    """在开发集或测试集上评估模型"""
    model.eval()  # 设置模型为评估模式 (关闭dropout等)
    predictions_list = []  # 存储所有批次的预测结果
    targets_list = []  # 存储所有批次的真实标签
    total_loss = 0

    with torch.no_grad():  # 评估时不需要计算梯度
        for data, target in tqdm(data_loader, desc='评估中', leave=False):
            data, target = data.to(device), target.to(device)  # 数据迁移到设备
            output, _ = model(data)  # 前向传播
            # 计算损失 (与训练时相同的方式)
            loss = criterion(output.permute(0, 2, 1), target)
            total_loss += loss.item()

            # 获取预测的类别索引 (在num_sentiment_classes维度上取最大值的索引)
            pred_indices = torch.argmax(output, dim=2)  # shape: (batch_size, num_aspects)
            predictions_list.extend(pred_indices.cpu().numpy())  # 转为numpy并保存
            targets_list.extend(target.cpu().numpy())

    predictions_np = np.array(predictions_list)  # shape: (num_samples, num_aspects)
    targets_np = np.array(targets_list)  # shape: (num_samples, num_aspects)

    metrics_by_class = {}  # 存储每个方面的评估指标
    if targets_np.size > 0 and predictions_np.shape == targets_np.shape:
        for i, aspect_name in enumerate(aspect_categories):
            y_true_aspect = targets_np[:, i]  # 当前方面的真实标签 (1D array)
            y_pred_aspect = predictions_np[:, i]  # 当前方面的预测标签 (1D array)
            # 调用多分类指标计算函数
            metrics_for_aspect = calculate_metrics_multiclass(y_true_aspect, y_pred_aspect, num_sentiment_classes)
            metrics_by_class[aspect_name] = metrics_for_aspect
    else:
        print("警告: 目标数组为空或形状不匹配，跳过详细指标计算。")
        # 如果无法计算，则填充默认值
        for aspect_name in aspect_categories:
            metrics_by_class[aspect_name] = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'specificity': 0,
                'confusion_matrix': [[0] * num_sentiment_classes for _ in range(num_sentiment_classes)]
            }

    # 计算所有方面的平均性能指标 (宏平均)
    avg_metrics = {
        metric_key: np.mean([m[metric_key] for m in metrics_by_class.values()]) if metrics_by_class else 0
        for metric_key in ['accuracy', 'precision', 'recall', 'f1', 'specificity']
    }

    print("\n--- 多分类情感评估结果 ---")
    print("注意: 每个方面的精确率, 召回率, F1, 特异度是基于其4个情感类别计算的宏平均值。")
    print("\n性能指标 (按具体方面细分)：")
    for aspect, metrics in metrics_by_class.items():
        print(f"\n{aspect}:")
        print(f"  准确率 (ACC): {metrics['accuracy']:.4f}")
        print(f"  精确率 (Macro-PPV): {metrics['precision']:.4f}")
        print(f"  召回率 (Macro-TPR): {metrics['recall']:.4f}")
        print(f"  F1分数 (Macro-F1): {metrics['f1']:.4f}")
        print(f"  特异度 (Macro-TNR): {metrics['specificity']:.4f}")
        print(f"  混淆矩阵:\n{np.array(metrics['confusion_matrix'])}")

    print("\n平均性能指标 (在各方面之间取平均)：")
    for key, value in avg_metrics.items():
        print(f"平均{key.capitalize()}: {value:.4f}")

    # 保存评估结果到文件
    output_dir = 'analysis_output'  # 结果输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 记录总的训练指标概览
    metrics_file_path = os.path.join(output_dir, 'training_summary_metrics.txt')  # 修改文件名以区分
    with open(metrics_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\nEpoch {current_epoch if current_epoch is not None else '最终评估'} (多分类):")
        f.write(f"Loss: {total_loss / len(data_loader) if len(data_loader) > 0 else float('inf'):.4f}\n")
        f.write(f"平均准确率 (ACC): {avg_metrics['accuracy']:.4f}\n")
        f.write(f"平均F1分数 (Macro-F1): {avg_metrics['f1']:.4f}\n")

    # 保存当前epoch的详细评估结果
    if current_epoch is not None:
        result_file_path = os.path.join(output_dir, f'epoch_{current_epoch + 1}_detailed_metrics_multiclass.txt')
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Epoch {current_epoch + 1} 多分类评估结果\n")
            f.write(f"\n整体平均指标 (所有方面平均):")
            for key, value in avg_metrics.items():
                f.write(f"  {key.capitalize()}: {value:.4f}\n")
            f.write("\n按方面细分的指标:")
            for aspect, metrics in metrics_by_class.items():
                f.write(f"\n  {aspect}:")
                for key, value in metrics.items():
                    if key != 'confusion_matrix':
                        f.write(f"    {key.capitalize()}: {value:.4f}\n")

    # 将当前epoch的metrics_by_class添加到历史记录，用于后续绘图
    if metrics_history is not None:
        metrics_history.append(metrics_by_class)

    return avg_metrics, total_loss / len(data_loader) if len(data_loader) > 0 else 0


# --- 主函数 (程序入口) ---
def main():
    """主执行函数"""
    # --- 设备配置 ---
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)  # 为所有GPU设置随机种子 (如果使用多卡)
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用 CPU 进行计算")

    # --- 加载数据 ---
    train_df = pd.read_csv('restaurant_comment_data/train.csv')
    test_df = pd.read_csv('restaurant_comment_data/test.csv')
    dev_df = pd.read_csv('restaurant_comment_data/dev.csv')

    # --- 数据增强 ---
    # 对训练数据进行增强，扩充训练集规模
    augmented_reviews = []
    augmented_labels_raw = []
    for idx, row in tqdm(train_df.iterrows(), desc='数据增强中'):
        aug_texts = augment_text(row['review'])
        for aug_text in aug_texts:
            augmented_reviews.append(aug_text)
            current_raw_labels = [int(row[aspect]) for aspect in aspect_categories]
            augmented_labels_raw.append(current_raw_labels)
    # 将增强数据构造成DataFrame并与原始训练数据合并
    aug_df_data = {'review': augmented_reviews}
    for i, aspect in enumerate(aspect_categories):
        aug_df_data[aspect] = [review_raw_labels[i] for review_raw_labels in augmented_labels_raw]
    aug_df = pd.DataFrame(aug_df_data)
    train_df = pd.concat([train_df, aug_df], ignore_index=True)
    print(f"数据增强后，训练集大小: {len(train_df)}")

    # --- 构建词汇表、数据集和数据加载器 ---
    vocab = build_vocab(train_df['review'])  # 基于增强后的训练集构建词汇表
    print(f"词汇表大小: {len(vocab)}")

    train_dataset = RestaurantDataset(train_df, vocab)
    dev_dataset = RestaurantDataset(dev_df, vocab)
    test_dataset = RestaurantDataset(test_df, vocab)

    # pin_memory=True 与 non_blocking=True 配合使用，可加速CUDA数据传输
    # num_workers > 0 使用多进程加载数据，提高效率
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True,
                              num_workers=4 if device.type == 'cuda' else 0)
    dev_loader = DataLoader(dev_dataset, batch_size=32, pin_memory=True, num_workers=4 if device.type == 'cuda' else 0)
    test_loader = DataLoader(test_dataset, batch_size=32, pin_memory=True, num_workers=4 if device.type == 'cuda' else 0)

    # --- 模型、损失函数、优化器、学习率调度器 初始化 ---
    model = EnhancedBiLSTMModel(
        vocab_size=len(vocab),
        embed_dim=300,  # 词向量维度
        hidden_dim=128,  # LSTM隐藏层维度
        num_aspects=len(aspect_categories),  # 方面类别数量
        num_sentiment_classes=num_sentiment_classes,  # 每个方面的情感类别数量
        dropout_rate=0.3
    ).to(device)  # 将模型迁移到指定设备

    # 使用交叉熵损失函数，适用于多分类任务
    criterion = nn.CrossEntropyLoss()

    # AdamW优化器，通常比Adam在NLP任务中表现更好，因其权重衰减方式不同
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # 初始学习率，OneCycleLR会动态调整它。对于AdamW和transformer类模型，较小的初始学习率通常更好。
        weight_decay=0.01  # 权重衰减，防止过拟合
    )

    # OneCycleLR学习率调度策略：先warmup增加学习率，然后余弦退火降低学习率
    num_epochs = 10  # 训练轮次
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-3,  # 学习率峰值
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # warmup阶段占总训练步数的比例 (例如10%的steps)
        anneal_strategy='cos',  # 余弦退火策略
        div_factor=100.0,  # 初始学习率 = max_lr / div_factor (例如 2e-3 / 100 = 2e-5)
        final_div_factor=1000.0  # 最低学习率 = 初始学习率 / final_div_factor
    )

    # --- 训练循环和评估 --- 
    # 定义模型保存路径
    output_dir = 'analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'best_model_on_f1.pth')  # 基于F1分数保存最佳模型

    grad_clip = 1.0  # 梯度裁剪阈值 (原为5.0，对于AdamW和较小学习率，1.0可能更合适)

    # 早停机制相关变量初始化
    early_stopping_patience = 3  # 如果连续3个epoch没有改善，则早停 (原为5)
    best_val_loss = float('inf')  # 记录最佳验证损失
    best_val_f1 = 0  # 记录最佳验证F1分数
    no_improve_count = 0  # 记录未改善的epoch计数

    train_losses = []  # 记录每个epoch的训练损失
    val_losses = []  # 记录每个epoch的验证损失
    metrics_history = []  # 记录每个epoch在验证集上的详细评估指标
    val_f1_scores = []  # 记录每个epoch在验证集上的平均F1分数

    print(f"\n开始训练，共 {num_epochs} 轮...")
    for epoch in range(num_epochs):
        print(f"\n--- 第 {epoch + 1}/{num_epochs} 轮 ---")
        # 训练模型
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler, grad_clip_value=grad_clip)
        train_losses.append(train_loss)
        # 在验证集上评估
        val_metrics, val_loss = evaluate(model, dev_loader, criterion, device, metrics_history, current_epoch=epoch)
        val_losses.append(val_loss)
        val_f1_scores.append(val_metrics['f1'])  # val_metrics['f1'] 是所有方面F1的平均值

        print(f"第 {epoch + 1} 轮结束: 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证平均F1: {val_metrics['f1']:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.6e}")  # 显示当前学习率

        # 早停检查逻辑：同时考虑验证损失和F1分数
        # 如果F1分数有提升，则保存模型并重置计数器
        # 如果F1没有提升但损失有降低，也认为有改善，重置计数器
        # 只有两者都无明显改善时，才增加no_improve_count
        improved_f1 = val_metrics['f1'] > best_val_f1
        improved_loss = val_loss < best_val_loss

        if improved_f1:
            print(f"验证集F1分数提升: {best_val_f1:.4f} -> {val_metrics['f1']:.4f}. 保存模型到 {best_model_path}")
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            no_improve_count = 0  # F1提升，重置计数
            if improved_loss:  # 如果损失也降低了，更新best_val_loss
                best_val_loss = val_loss
        elif improved_loss:  # F1未提升，但损失降低
            print(f"验证集损失降低: {best_val_loss:.4f} -> {val_loss:.4f}. (F1未提升)")
            best_val_loss = val_loss
            no_improve_count = 0  # 损失降低，也认为有进展，重置计数
        else:  # F1和Loss均未改善
            no_improve_count += 1
            print(f"验证集指标无明显改善 ({no_improve_count}/{early_stopping_patience})")

        if no_improve_count >= early_stopping_patience:
            print(f'触发早停机制于第 {epoch + 1} 轮。')
            break  # 跳出训练循环

    # --- 训练后处理：绘图和最终测试 ---
    print("\n训练完成或早停。开始生成图表...")
    # 动态导入绘图函数，避免在无显示环境的服务器上因matplotlib后端问题报错 (如果适用)
    try:
        from plot_epoch_metrics import plot_aspect_metrics_for_epoch, \
            plot_training_and_f1_curves, \
            plot_individual_metric_trends

        # 绘制训练/验证损失曲线和F1分数曲线
        if train_losses and val_losses and val_f1_scores:
            plot_training_and_f1_curves(train_losses, val_losses, val_f1_scores, output_dir=output_dir)
        else:
            print("警告: 数据不足，无法绘制训练和F1曲线。")

        # 绘制各平均指标随epoch变化的趋势图
        if metrics_history:
            plot_individual_metric_trends(metrics_history, output_dir=output_dir)
        else:
            print("警告: 数据不足，无法绘制指标趋势图。")
        print("图表已保存到 analysis_output 目录。")
    except ImportError:
        print("警告: plot_epoch_metrics.py 未找到或 matplotlib/seaborn 未安装，跳过绘图。")
    except Exception as e:
        print(f"绘图时发生错误: {e}")

    # 加载在验证集上F1表现最佳的模型，并在测试集上进行最终评估
    if os.path.exists(best_model_path):
        print(f"\n加载在验证集上F1最佳的模型从: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print(f"警告: 未找到最佳模型文件于 {best_model_path}。将使用训练结束时的模型进行测试。")

    print("\n在测试集上进行最终评估...")
    test_metrics, test_loss = evaluate(model, test_loader, criterion, device)
    print(f"测试集损失: {test_loss:.4f}")
    # test_metrics 包含了 'accuracy', 'precision', 'recall', 'f1', 'specificity' 的平均值
    print("\n测试集平均性能指标 (所有方面取平均)：")
    for key, value in test_metrics.items():
        print(f"  平均{key.capitalize()}: {value:.4f}")
    print("\n=== 分析完成 ===")


if __name__ == '__main__':
    main()
