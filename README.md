# 餐饮评论细粒度情感分析

## 项目概述
本项目实现了一个基于BiLSTM和注意力机制的细粒度情感分析模型，用于分析餐饮评论中的多个方面（如位置、服务、价格、环境、食物等）的情感极性。模型能够识别四种不同的情感状态：正面(1)、中性(0)、负面(-1)和未提及(-2)，提供更加细粒度的情感分析结果。

## 环境要求
确保您的环境中安装了以下库：

- Python 3.8+
- PyTorch (>= 1.8.0 推荐)
- pandas
- numpy
- jieba (用于中文分词)
- scikit-learn (用于评估指标计算)
- matplotlib (用于绘图)
- seaborn (用于绘图)
- tqdm (用于显示进度条)

您可以使用 pip 来安装这些依赖：
```bash
pip install torch pandas numpy jieba scikit-learn matplotlib seaborn tqdm
```
*注意：请根据您的系统和CUDA版本安装合适的PyTorch版本。* 

## 数据集
数据集包含餐饮评论及其对应的18个细粒度方面的情感标签：
- 位置相关：交通、市中心位置、易找程度
- 服务相关：排队、服务态度、停车、及时性
- 价格相关：价格水平、性价比、折扣
- 环境相关：装修、噪音、空间、卫生
- 食物相关：分量、口味、外观、推荐度

情感标签采用四级标注：
- 1：正面评价
- 0：中性评价
- -1：负面评价
- -2：未提及该方面

数据集分为：
- train.csv：训练集
- dev.csv：验证集
- test.csv：测试集

## 使用说明

1.  **配置环境**：按照“环境要求”部分安装所需库。
2.  **运行分析**: 在项目根目录下执行以下命令：
    ```bash
    python analysis.py
    ```
3.  **查看输出**: 
    *   训练过程日志（包括每个epoch的损失、指标、早停信息等）将打印到控制台。
    *   最佳模型（基于验证集F1分数）将保存为 `analysis_output/best_model_on_f1.pth`。
    *   评估指标的文本日志（每轮的详细指标、训练总结）将保存在 `analysis_output` 目录下。
    *   性能可视化图表（损失/F1曲线、指标趋势图等）将保存在 `analysis_output` 目录下。

## 注意事项
- 建议使用支持 CUDA 的 GPU 进行训练以显著提高效率。脚本会自动检测并使用GPU（如果可用）。
- 训练时间和所需资源取决于您的硬件配置和数据集大小。

## 文件说明
- [analysis.py](analysis.py)：模型程序文件
- [plot_epoch_metrics.py](plot_epoch_metrics.py)：绘制训练曲线的工具函数
- [Model_details.md](Model_details.md)：模型详解
- [README.md](README.md)：项目运行说明文档