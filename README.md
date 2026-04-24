# 紫砂壶特征识别多任务网络 (ChaHu Multi-Task Net)

本项目基于 **Backbone-Neck-Head** 解耦设计，实现了一个针对紫砂壶图像的多任务分类模型。模型能够同时识别紫砂壶的**花型 (Flower Type)** 和 **提手类型 (Handle Type)**，并采用了基于 Mask 的背景扣除预处理技术。

## 一、项目结构

```c++
Project/
├── chahu_model_epoch_x.pth/            # 存放训练好的模型权重 (.pth 文件)
│                   # 核心代码模块 (可选，也可直接放在根目录)
├── model.py            # 模型架构 (Backbone, Neck, Head)
├── utils.py            # 数据处理、Dataset类、预处理逻辑
├──  evaluation.py       # 评估指标计算逻辑
│
├──train.py                # 主训练脚本
├── predict.py              # 推理/预测脚本
├── debug.py                # 环境诊断与调试脚本
│
├── requirements.txt        # 项目依赖库列表
└── README.md               # 项目说明文档
```




## 二、模型架构
### 1.总览

本项目将网络拆解为三个核心模块：
① Backbone (骨干网)：以resnet18为基础，负责从原始图片中逐层提取出从边缘纹理到复杂形状的通用视觉特征。

② Neck (颈部)：负责将繁杂的高维特征图进行压缩与降维，提炼出高度浓缩的核心特征向量。

③ Head (头部)：负责接收浓缩后的特征向量，并针对如分类判断进行最终的数学计算与决策输出。

```python
class Backbone(nn.Module):
    """
    【骨干网络】：负责从原始图像中提取高维特征图。
    """
    def __init__(self):
        super().__init__()
        # 加载预训练权重，利用 ImageNet 的先验知识识别纹理
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 取出除去最后两层（全局池化和全连接）的所有层
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 512 # ResNet18 最后一层特征图的深度

    def forward(self, x):
        return self.features(x) # 输出: [Batch, 512, 7, 7] (对于 224 输入)

class Neck(nn.Module):
    """
    【颈部网络】：负责对特征图进行空间降维和全局特征整合。
    """
    def __init__(self, in_channels):
        super().__init__()
        # 全局平均池化：将 7x7 的空间维度压缩为 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.avgpool(x)
        return torch.flatten(x, 1) # 输出: [Batch, 512]

class ClassificationHead(nn.Module):
    """
    【分类头】：负责根据整合后的特征进行最终的类别概率预测。
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256), # 标准化，加速收敛并防止梯度爆炸
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),     # 丢弃 30% 神经元，防止模型对 Else 类过拟合
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class MultiTaskNet(nn.Module):
    """
    【主网络】：整合 Backbone, Neck 和两个独立的 Head。
    实现“一套特征，多项任务”的并行预测。
    """
    def __init__(self, num_flower_classes, num_handle_classes):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(self.backbone.out_channels)
        # 分别定义花型和提手的预测分支
        self.flower_head = ClassificationHead(self.backbone.out_channels, num_flower_classes)
        self.handle_head = ClassificationHead(self.backbone.out_channels, num_handle_classes)

    def forward(self, x):
        feat = self.backbone(x)         # 提取特征图
        shared_feat = self.neck(feat)   # 整合为特征向量
        # 输出两组预测概率分布（Logits）
        return self.flower_head(shared_feat), self.handle_head(shared_feat)
```
**以下为模型结构图**
<img width="333" height="401" alt="image" src="https://github.com/user-attachments/assets/29a880f8-4d0d-483c-81fd-b5e27ecb2eac" />

 ### 2.Backbone
 <img width="485" height="934" alt="image" src="https://github.com/user-attachments/assets/e9f32c93-3daf-40b7-8c7b-95d7fed7385e" />
 义resnet18为基础
backbone 的逻辑：
 
1. 初始下采样：通过第一层 7x7 卷积和最大池化，迅速降低图像空间分辨率，提取基础的纹理和边缘特征。

2. 残差学习阶段 (Layer 1-4)：通过 4 组由残差块（Residual Blocks）组成的阶段，在保持梯度传递稳定的同时，提取越来越复杂的几何形状和语义信息。

3. 空间与通道的平衡：随着网络加深，特征图的空间尺寸（长宽）每进入一个新阶段就会缩小一半（下采样），而特征通道数（深度）则会翻倍，实现了信息的极度浓缩。
 ###3.Backbone
 ### 3.Neck
 使用全局平均池化 (Global Average Pooling, GAP)，将 7x7 的空间维度压缩为 1x1
 ### 4.Head
 <img width="1521" height="82" alt="Mermaid" src="https://github.com/user-attachments/assets/f2921068-a23b-4f89-a023-67a664276a49" />
 包含两个独立的分类头：`Flower Head`（17类花型）与 `Handle Head`（6类提手）。它们结构相同但权重独立，各自专注于自己的专业领域为了防止模型在长尾分布的紫砂壶数据集上过拟合，我们在分类头中引入了标准的“信息漏斗”范式（MLP级联结构），而非单一的全连接层。
 
逻辑：

1. 特征提纯 (Linear 512->256):
颈部传来的 512 维特征是“花型”和“提手”特征的混合体。此层通过矩阵乘法，过滤掉与当前任务无关的冗余信息，提取出 256 个最核心的专属特征。

2. 数据稳压 (BatchNorm1d):
强制将隐藏层激活值拉回标准正态分布，有效缓解内部协变量偏移（Internal Covariate Shift），使模型在面对罕见类别时收敛得更稳、更快。

3. 引入非线性 (ReLU):
打破线性堆叠的局限，赋予网络拟合紫砂壶复杂且扭曲的分类边界的能力。

4. 对抗过拟合 (Dropout 0.3):
在训练时随机“失活” 30% 的神经元。这逼迫模型不能仅依赖单一特征（如只看壶嘴），而是综合全局信息进行决策，极大提升了模型在测试集上的泛化能力。

5. 最终投票 (Linear 256->num_classes):
将提纯后的高级语义特征，映射为目标任务的原始得分向量（Logits），随后交由交叉熵损失函数（CrossEntropyLoss）进行评估。
### 5.模型创新
本项目并非简单地调用开箱即用的分类网络，而是针对紫砂壶数据集“样本量有限”、“存在长尾分布”、“背景复杂”等痛点，在架构和工程上进行了深度定制与创新：

1. 基于“硬参数共享”的多任务联合学习
传统的做法是针对花型和提手分别训练两个独立的模型，这不仅会导致算力与显存的双倍消耗，还忽略了两者间的结构关联。
- **特征协同**：本项目采用单 Backbone 双 Head 的多任务架构。在反向传播时，花型损失与提手损失共同指导底层权重的更新。
- **隐式正则化**：这种联合优化机制迫使骨干网络提取出更为泛化、全局的紫砂壶特征，极大地降低了模型在小规模数据集上的过拟合风险，实现了**“1+1>2”**的特征协同效应。

2. 基于 Mask 的先验注意力引导 
紫砂壶的图片背景往往极其复杂（茶盘、茶宠、杂乱的桌面），容易造成模型注意力偏移。
- **背景抑制**：在数据预处理阶段，我们在 `utils.py` 中引入了原图与 Mask 的按位乘法融合（Masked Imageing）。
- **强制聚焦**：在不增加网络额外计算开销的前提下，从物理层面抠除背景干扰，强制模型的卷积核 100% 聚焦于壶身的纹理与几何特征上。

3.针对长尾分布的防御性 Head 设计
鉴于实际数据集中“Else”等少数类别极不平衡（典型的长尾分布），我们抛弃了简单的单层线性分类器，重新设计了具备防御能力的 MLP 分类头。
- **稳压与提纯**：在输出 Logits 前，引入了隐层特征提纯（256维）与 `BatchNorm1d`，有效缓解内部协变量偏移（Internal Covariate Shift），加速长尾类别收敛。
- **随机失活**：引入 `Dropout(0.3)` 破坏神经元间的共适应性，逼迫模型不能仅依赖单一高频特征（如壶嘴），而是必须综合全局信息进行综合投票。

4. 极致解耦的“高内聚低耦合”工程架构
模型代码 (`model.py`) 严格遵循 Backbone - Neck - Head 的工业级分层范式：
- **Backbone**：采用带有残差连接的轻量级 ResNet18，保障足够感受野的同时避免梯度消失。
- **Neck**：利用 `AdaptiveAvgPool2d` 实现空间特征到一维语义特征的转换，消灭了全连接带来的海量参数爆炸，并赋予模型强大的**空间平移不变性**。
- **高可扩展性**：若未来需要增加新任务，只需极低成本的代码改动即可实现无缝热拔插。

## 三、数据预处理
在真实的紫砂壶图像中，往往充斥着极其复杂的背景干扰（如茶盘、茶宠、杂乱的桌面），且开源数据集常常伴随着样本不均衡和脏数据等问题。为此，本项目在 `utils.py` 中进行数据预处理。

1. 稀有类别安全合并 
为了防止在后续分层采样时因样本极少而导致报错（如训练集中全无某类，而测试集中存在），我们在加载数据后进行了安全拦截：
- 自动统计 `flower type` 类别分布。
- 将**样本数不足 10 的罕见类别**（排除本身就是 Else/Other 的类别）自动合并至统一的 `'Else'` 类中。这保证了底层采样逻辑的安全稳定。

2. 科学的 8:1:1 分层抽样
针对合并后的长尾分布，采用两次连续的 `train_test_split` 策略，严格按照类别比例划分：
- **第一次切分**：提取 80% 作为训练集，剩下 20% 作为临时集。
- **第二次切分**：将临时集对半分为 10% 验证集（Val）和 10% 测试集（Test）。
- 全程开启 `stratify_by_column="flower type"` 与固定种子 `seed=42`，确保每一次运行，罕见壶型都能科学地分布在三个数据集中。

3. 尺寸修正与背景抠除 
在 PyTorch 的 `Dataset` 读取阶段 (`ChaHuDataset`)，为了让模型聚焦于紫砂壶本身的几何与纹理：
- **尺寸修正**：当发现原图（RGB）与灰度蒙版（Mask）存在微小尺寸差异时，强制使用 `Image.Resampling.NEAREST`（最近邻插值）对 Mask 进行缩放。这样既对齐了 Numpy 矩阵以防广播失败，又不会让蒙版边缘变模糊。
- **矩阵乘法抠图**：将归一化后的蒙版与原图进行按位乘法 `(img * mask[..., None])`，从物理层面精准消除了杂乱桌面的背景干扰。

4. 动态标签编码与张量标准化
- 使用 Datasets 库自带的 `class_encode_column` 自动完成花型和提手字符串到数字 ID 的映射。
- 进入网络前，使用标准的 torchvision 处理管线：将所有图片统一 `Resize` 至 224x224，转换为 `Tensor`，并应用 ImageNet 的全局均值和方差 `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`，以此最大化榨取 ResNet18 预训练权重的先验知识。

## 四、训练过程
1. 硬件与计算环境
为了最大化训练效率，系统在启动时会自动检测硬件环境：
- **设备加速**：优先调用 NVIDIA CUDA 核心进行并行计算；若未检测到 GPU，系统将平滑降级至 CPU 模式。
- **动态日志**：引入 `flush=True` 机制，确保在 Windows 或远程环境下也能实时获取训练日志，防止程序崩溃导致的日志丢失。

2. 超参数配置 (Hyperparameters)
训练方案基于工业级经验进行预设，以保障收敛的稳定性：
- **优化器**: Adam (Adaptive Moment Estimation)
- **学习率 (LR)**: 1e-4 (兼顾收敛速度与精细化微调)
- **批大小 (Batch Size)**: 32
- **迭代轮数 (Epochs)**: 10
- **输入尺寸**: 224 × 224 (符合 ResNet18 标准感受野)

3. 多任务联合损失函数 (Multi-Task Loss)
本项目训练的核心在于**任务协同优化**。在每一轮前向传播中，模型会同时输出两个分支的预测结果。
总损失函数公式定义为：
$$L_{total} = L_{flower\_type} + L_{handle\_type}$$
其中 $L$ 均为 **CrossEntropyLoss（交叉熵损失）**。通过将两个任务的梯度在 Neck 层进行聚合，强迫骨干网络学习到更具泛化性的紫砂壶底层特征。

4. 实时监控与闭环评估
- **可视化进度**：集成 `tqdm` 进度条，实时显示 Batch 级别的 Loss 波动。
- **每轮验证 (Per-Epoch Validation)**：在每个 Epoch 结束后，系统立即调用 `evaluation.py` 在验证集（Validation Set）上运行推理。通过计算 **Precision（精确率）**、**Recall（召回率）** 以及针对不平衡数据的 **Macro F1-Score**，实时监控模型是否出现过拟合。

5. 模型持久化与最终评估
- **自动保存**：每轮训练结束后，系统会自动保存带有轮数标记的权重文件（如 `chahu_model_epoch_10.pth`）。
- **终极考核**：全部训练结束后，系统将自动加载验证集表现最佳的权重，在**完全隔离的测试集（Test Set）**上进行最终评估，确保模型具备真实的业务推理能力。

6. 训练日志
见
