import torch
import torch.nn as nn
from torchvision import models

class Backbone(nn.Module):
    """
    【骨干网络】：负责从原始图像中提取高维特征图。
    这里使用 ResNet18 的卷积层，剔除了全连接层。
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