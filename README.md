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




##二、模型架构
###1.总览

本项目将网络拆解为三个核心模块：
① Backbone (骨干网)：负责从原始图片中逐层提取出从边缘纹理到复杂形状的通用视觉特征。

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

 ###2.Backbone
 <img width="485" height="934" alt="image" src="https://github.com/user-attachments/assets/e9f32c93-3daf-40b7-8c7b-95d7fed7385e" />
 **Backbone 的逻辑：**
初始下采样：通过第一层 7x7 卷积和最大池化，迅速降低图像空间分辨率，提取基础的纹理和边缘特征。

残差学习阶段 (Layer 1-4)：通过 4 组由残差块（Residual Blocks）组成的阶段，在保持梯度传递稳定的同时，提取越来越复杂的几何形状和语义信息。

空间与通道的平衡：随着网络加深，特征图的空间尺寸（长宽）每进入一个新阶段就会缩小一半（下采样），而特征通道数（深度）则会翻倍，实现了信息的极度浓缩。
 ###3.Backbone
 



## 项目依赖库列表

```c++
# 核心深度学习框架
torch
torchvision

# 数据集处理与加载
datasets
pyarrow 

# 图像与矩阵运算
Pillow
numpy

# 模型评估与训练可视化
scikit-learn
tqdm
```

## train.py运行结果

```c++
D:\ProgrammePython\Anaconda\envs\Project01\python.exe E:\Project\train.py 
==================================================
 初始化训练环境...
==================================================
 当前计算设备: GPU (NVIDIA GeForce GTX 1650)

 正在获取并处理数据集... 
 开始加载数据集 split='EN'...
 数据集加载成功，原始样本总数: 9753
 正在统计 'flower type' 分布...
警告: 发现 15 个稀有类别样本数不足 10，正在合并至 'Else'...
被合并的类别包括: ['瓜棱壶', '软提梁壶', '僧帽壶', '足鼎壶', '寿星壶']... 等
正在执行标签数字编码 (class_encode_column)...
正在执行分层采样划分 (Stratified Split)...

数据集划分完成:
   ├─训练集 (Train): 7802 样本
   ├─验证集 (Val):   975 样本
   └─测试集 (Test):  976 样本

 数据集真实类别统计:
   ├─ 花型 (Flower): 17 类
   └─ 提手 (Handle): 6 类

 正在封装 DataLoader...
PyTorch Dataset 实例已创建，包含 7802 个样本。
PyTorch Dataset 实例已创建，包含 975 个样本。

 正在构建 MultiTaskNet 模型与优化器...
 模型初始化完成。

==================================================
开始模型训练 (共计 10 Epochs)
==================================================
                                                                                  
 Epoch [1/10] 训练完成 | 训练集平均 Loss: 2.1645
正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       1.00      0.11      0.20         9
           1       0.00      0.00      0.00        10
           2       0.00      0.00      0.00         3
           3       0.00      0.00      0.00         5
           4       0.50      0.87      0.63        15
           5       0.00      0.00      0.00        10
           6       0.33      0.12      0.18        16
           7       0.00      0.00      0.00         1
           8       0.91      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       0.00      0.00      0.00         7
          11       0.44      0.89      0.59         9
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         2
          14       0.00      0.00      0.00         3
          15       0.45      0.17      0.25        29
          16       0.00      0.00      0.00         1

    accuracy                           0.89       975
   macro avg       0.21      0.18      0.16       975
weighted avg       0.84      0.89      0.85       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.73      0.73      0.73        30
           1       0.99      1.00      0.99       918
           2       0.78      0.47      0.58        15
           3       0.62      0.42      0.50        12

    accuracy                           0.97       975
   macro avg       0.78      0.65      0.70       975
weighted avg       0.97      0.97      0.97       975

Epoch [2/10]:   0%|          | 0/244 [00:00<?, ?it/s] 模型已保存至: chahu_model_epoch_1.pth
--------------------------------------------------
                                                                                  
 Epoch [2/10] 训练完成 | 训练集平均 Loss: 0.7235
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         9
           1       0.00      0.00      0.00        10
           2       0.00      0.00      0.00         3
           3       0.00      0.00      0.00         5
           4       1.00      0.47      0.64        15
           5       0.33      0.10      0.15        10
           6       0.10      0.06      0.08        16
           7       0.00      0.00      0.00         1
           8       0.92      0.94      0.93       851
           9       0.00      0.00      0.00         3
          10       0.00      0.00      0.00         7
          11       0.62      0.56      0.59         9
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         2
          14       0.00      0.00      0.00         3
          15       0.19      0.48      0.27        29
          16       0.00      0.00      0.00         1

    accuracy                           0.85       975
   macro avg       0.19      0.15      0.16       975
weighted avg       0.84      0.85      0.84       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.89      0.80      0.84        30
           1       0.99      1.00      0.99       918
           2       0.80      0.53      0.64        15
           3       0.67      0.50      0.57        12

    accuracy                           0.98       975
   macro avg       0.84      0.71      0.76       975
weighted avg       0.98      0.98      0.98       975

 模型已保存至: chahu_model_epoch_2.pth
--------------------------------------------------
                                                                                  
 Epoch [3/10] 训练完成 | 训练集平均 Loss: 0.4795
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       1.00      0.11      0.20         9
           1       0.00      0.00      0.00        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       1.00      0.40      0.57        15
           5       0.43      0.30      0.35        10
           6       0.00      0.00      0.00        16
           7       0.00      0.00      0.00         1
           8       0.91      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       0.00      0.00      0.00         7
          11       0.60      0.67      0.63         9
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         2
          14       0.00      0.00      0.00         3
          15       0.50      0.48      0.49        29
          16       0.00      0.00      0.00         1

    accuracy                           0.89       975
   macro avg       0.32      0.18      0.21       975
weighted avg       0.85      0.89      0.86       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.95      0.63      0.76        30
           1       0.99      1.00      0.99       918
           2       0.64      0.60      0.62        15
           3       0.57      0.67      0.62        12

    accuracy                           0.98       975
   macro avg       0.79      0.72      0.75       975
weighted avg       0.98      0.98      0.98       975

Epoch [4/10]:   0%|          | 0/244 [00:00<?, ?it/s] 模型已保存至: chahu_model_epoch_3.pth
--------------------------------------------------
                                                                                  
 Epoch [4/10] 训练完成 | 训练集平均 Loss: 0.3174
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       0.50      0.22      0.31         9
           1       0.00      0.00      0.00        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       1.00      0.67      0.80        15
           5       0.60      0.30      0.40        10
           6       0.00      0.00      0.00        16
           7       0.00      0.00      0.00         1
           8       0.91      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       0.00      0.00      0.00         7
          11       0.88      0.78      0.82         9
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         2
          14       0.00      0.00      0.00         3
          15       0.33      0.38      0.35        29
          16       0.00      0.00      0.00         1

    accuracy                           0.89       975
   macro avg       0.31      0.21      0.23       975
weighted avg       0.85      0.89      0.87       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.92      0.73      0.81        30
           1       0.99      1.00      0.99       918
           2       0.64      0.60      0.62        15
           3       0.70      0.58      0.64        12

    accuracy                           0.98       975
   macro avg       0.81      0.73      0.77       975
weighted avg       0.98      0.98      0.98       975

Epoch [5/10]:   0%|          | 0/244 [00:00<?, ?it/s] 模型已保存至: chahu_model_epoch_4.pth
--------------------------------------------------
                                                                                  
 Epoch [5/10] 训练完成 | 训练集平均 Loss: 0.2071
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       0.40      0.22      0.29         9
           1       0.00      0.00      0.00        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       0.92      0.80      0.86        15
           5       0.40      0.40      0.40        10
           6       0.25      0.12      0.17        16
           7       0.00      0.00      0.00         1
           8       0.92      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       1.00      0.29      0.44         7
          11       0.78      0.78      0.78         9
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         2
          14       1.00      0.67      0.80         3
          15       0.61      0.38      0.47        29
          16       0.00      0.00      0.00         1

    accuracy                           0.90       975
   macro avg       0.43      0.28      0.32       975
weighted avg       0.87      0.90      0.88       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.86      0.63      0.73        30
           1       0.99      1.00      0.99       918
           2       0.56      0.60      0.58        15
           3       0.50      0.50      0.50        12

    accuracy                           0.97       975
   macro avg       0.73      0.68      0.70       975
weighted avg       0.97      0.97      0.97       975

Epoch [6/10]:   0%|          | 0/244 [00:00<?, ?it/s] 模型已保存至: chahu_model_epoch_5.pth
--------------------------------------------------
                                                                                  
Epoch [6/10] 训练完成 | 
    训练集平均 Loss: 0.1214 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       0.40      0.22      0.29         9
           1       0.33      0.20      0.25        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       0.92      0.73      0.81        15
           5       0.33      0.30      0.32        10
           6       0.25      0.25      0.25        16
           7       0.00      0.00      0.00         1
           8       0.92      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       0.33      0.14      0.20         7
          11       1.00      0.44      0.62         9
          12       1.00      1.00      1.00         1
          13       0.00      0.00      0.00         2
          14       1.00      0.67      0.80         3
          15       0.79      0.38      0.51        29
          16       0.00      0.00      0.00         1

    accuracy                           0.90       975
   macro avg       0.49      0.32      0.37       975
weighted avg       0.88      0.90      0.88       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.91      0.67      0.77        30
           1       0.99      1.00      0.99       918
           2       0.57      0.53      0.55        15
           3       0.55      0.50      0.52        12

    accuracy                           0.97       975
   macro avg       0.75      0.67      0.71       975
weighted avg       0.97      0.97      0.97       975

Epoch [7/10]:   0%|          | 0/244 [00:00<?, ?it/s] 模型已保存至: chahu_model_epoch_6.pth
--------------------------------------------------
                                                                                  
 Epoch [7/10] 训练完成 | 训练集平均 Loss: 0.0790
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       1.00      0.11      0.20         9
           1       0.25      0.10      0.14        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       1.00      0.73      0.85        15
           5       0.43      0.30      0.35        10
           6       0.20      0.06      0.10        16
           7       0.00      0.00      0.00         1
           8       0.91      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       0.50      0.29      0.36         7
          11       0.83      0.56      0.67         9
          12       1.00      1.00      1.00         1
          13       1.00      0.50      0.67         2
          14       0.67      0.67      0.67         3
          15       0.53      0.31      0.39        29
          16       0.00      0.00      0.00         1

    accuracy                           0.90       975
   macro avg       0.55      0.34      0.39       975
weighted avg       0.87      0.90      0.87       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.92      0.73      0.81        30
           1       0.99      1.00      0.99       918
           2       0.73      0.53      0.62        15
           3       0.55      0.50      0.52        12

    accuracy                           0.98       975
   macro avg       0.79      0.69      0.74       975
weighted avg       0.98      0.98      0.98       975

 模型已保存至: chahu_model_epoch_7.pth
--------------------------------------------------
                                                                                  
 Epoch [8/10] 训练完成 | 训练集平均 Loss: 0.0581
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       0.40      0.22      0.29         9
           1       0.00      0.00      0.00        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       0.82      0.60      0.69        15
           5       0.42      0.50      0.45        10
           6       0.50      0.06      0.11        16
           7       0.00      0.00      0.00         1
           8       0.91      0.98      0.95       851
           9       0.00      0.00      0.00         3
          10       0.67      0.29      0.40         7
          11       0.86      0.67      0.75         9
          12       1.00      1.00      1.00         1
          13       1.00      0.50      0.67         2
          14       0.67      0.67      0.67         3
          15       0.64      0.24      0.35        29
          16       0.00      0.00      0.00         1

    accuracy                           0.90       975
   macro avg       0.52      0.35      0.39       975
weighted avg       0.87      0.90      0.87       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.84      0.70      0.76        30
           1       0.99      0.99      0.99       918
           2       0.56      0.67      0.61        15
           3       0.62      0.67      0.64        12
           4       0.00      0.00      0.00         0

    accuracy                           0.97       975
   macro avg       0.60      0.60      0.60       975
weighted avg       0.98      0.97      0.97       975

 模型已保存至: chahu_model_epoch_8.pth
--------------------------------------------------
                                                                                  
 Epoch [9/10] 训练完成 | 训练集平均 Loss: 0.0401
 正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       1.00      0.11      0.20         9
           1       1.00      0.10      0.18        10
           2       0.00      0.00      0.00         3
           3       1.00      0.20      0.33         5
           4       0.83      0.67      0.74        15
           5       0.50      0.30      0.38        10
           6       0.33      0.06      0.11        16
           7       0.00      0.00      0.00         1
           8       0.91      0.99      0.95       851
           9       0.00      0.00      0.00         3
          10       0.50      0.14      0.22         7
          11       0.83      0.56      0.67         9
          12       1.00      1.00      1.00         1
          13       0.00      0.00      0.00         2
          14       1.00      0.67      0.80         3
          15       0.67      0.34      0.45        29
          16       0.00      0.00      0.00         1

    accuracy                           0.90       975
   macro avg       0.56      0.30      0.35       975
weighted avg       0.88      0.90      0.87       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.90      0.63      0.75        30
           1       0.99      1.00      0.99       918
           2       0.64      0.60      0.62        15
           3       0.62      0.67      0.64        12

    accuracy                           0.98       975
   macro avg       0.79      0.72      0.75       975
weighted avg       0.98      0.98      0.98       975

模型已保存至: chahu_model_epoch_9.pth
--------------------------------------------------
                                                                                   
Epoch [10/10] 训练完成 | 训练集平均 Loss: 0.0386
正在验证集上评估泛化能力...

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       1.00      0.11      0.20         9
           1       0.11      0.10      0.11        10
           2       0.00      0.00      0.00         3
           3       0.50      0.20      0.29         5
           4       0.77      0.67      0.71        15
           5       0.00      0.00      0.00        10
           6       1.00      0.06      0.12        16
           7       0.00      0.00      0.00         1
           8       0.92      0.98      0.95       851
           9       0.50      0.33      0.40         3
          10       0.67      0.29      0.40         7
          11       0.89      0.89      0.89         9
          12       1.00      1.00      1.00         1
          13       0.00      0.00      0.00         2
          14       1.00      0.33      0.50         3
          15       0.57      0.45      0.50        29
          16       0.00      0.00      0.00         1

    accuracy                           0.90       975
   macro avg       0.52      0.32      0.36       975
weighted avg       0.88      0.90      0.87       975


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.82      0.77      0.79        30
           1       0.99      1.00      0.99       918
           2       0.58      0.47      0.52        15
           3       0.70      0.58      0.64        12

    accuracy                           0.98       975
   macro avg       0.77      0.70      0.74       975
weighted avg       0.97      0.98      0.98       975

模型已保存至: chahu_model_epoch_10.pth
--------------------------------------------------

全部训练结束！
 正在使用测试集进行最终评估...
PyTorch Dataset 实例已创建，包含 976 个样本。

[Flower Type 分类报告]
              precision    recall  f1-score   support

           0       0.67      0.22      0.33         9
           1       0.31      0.36      0.33        11
           2       0.50      0.33      0.40         3
           3       0.00      0.00      0.00         4
           4       0.91      0.67      0.77        15
           5       1.00      0.10      0.18        10
           6       0.71      0.29      0.42        17
           7       0.00      0.00      0.00         1
           8       0.92      0.98      0.95       851
           9       0.50      0.33      0.40         3
          10       1.00      0.12      0.22         8
          11       0.80      0.44      0.57         9
          12       0.00      0.00      0.00         1
          13       0.00      0.00      0.00         1
          14       0.00      0.00      0.00         2
          15       0.56      0.33      0.42        30
          16       0.00      0.00      0.00         1

    accuracy                           0.90       976
   macro avg       0.46      0.25      0.29       976
weighted avg       0.88      0.90      0.88       976


[Handle Type 分类报告]
              precision    recall  f1-score   support

           0       0.76      0.85      0.80        40
           1       0.99      1.00      0.99       897
           2       0.67      0.43      0.53        23
           3       0.82      0.56      0.67        16

    accuracy                           0.97       976
   macro avg       0.81      0.71      0.75       976
weighted avg       0.97      0.97      0.97       976


进程已结束，退出代码为 0

```

