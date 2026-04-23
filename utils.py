import numpy as np
import torch
from PIL import Image
from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset


def prepare_data(split_name='EN'):
    """
    加载并安全划分数据集。
    针对极少样本类别进行合并，确保分层采样（Stratify）不报错。
    """
    print(f" 开始加载数据集 split='{split_name}'...")
    ds = load_dataset("AGI-FBHC/ChaHu", split=split_name)
    print(f" 数据集加载成功，原始样本总数: {len(ds)}")

    # 1. 类别统计与合并
    print(" 正在统计 'flower type' 分布...")
    counts = Counter(ds['flower type'])

    # 找出样本数不足 10 的类别
    rare = {k for k, v in counts.items() if v < 10 and k not in ['Else', 'Other']}

    if rare:
        print(f"警告: 发现 {len(rare)} 个稀有类别样本数不足 10，正在合并至 'Else'...")
        print(f"被合并的类别包括: {list(rare)[:5]}... 等")  # 仅展示前5个
        ds = ds.map(lambda x: {'flower type': 'Else' if x['flower type'] in rare else x['flower type']},
                    desc="正在应用类别合并")
    else:
        print("未发现需要合并的极端稀有类别。")

    # 2. 标签编码
    print("正在执行标签数字编码 (class_encode_column)...")
    ds = ds.class_encode_column("flower type")
    ds = ds.class_encode_column("handle type")

    # 3. 两次分层划分 (8:1:1)
    print("正在执行分层采样划分 (Stratified Split)...")

    # 划分出 20% 作为临时集 (临时集 = 验证 + 测试)
    train_temp = ds.train_test_split(test_size=0.2, stratify_by_column="flower type", seed=42)

    # 将 20% 的临时集平分为 10% 验证集和 10% 测试集
    val_test = train_temp['test'].train_test_split(test_size=0.5, stratify_by_column="flower type", seed=42)

    train_ds = train_temp['train']
    val_ds = val_test['train']
    test_ds = val_test['test']

    print(f"\n数据集划分完成:")
    print(f"   训练集 (Train): {len(train_ds)} 样本")
    print(f"   验证集 (Val):   {len(val_ds)} 样本")
    print(f"   测试集 (Test):  {len(test_ds)} 样本\n")

    return train_ds, val_ds, test_ds


class ChaHuDataset(Dataset):
    """
    PyTorch 封装类：在读取每一张图时动态应用 Mask 背景扣除。
    """

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        # 在初始化时静默检查一次
        if len(hf_dataset) > 0:
            print(f"PyTorch Dataset 实例已创建，包含 {len(hf_dataset)} 个样本。")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 1. 取出 PIL Image 对象
        img_pil = item['image'].convert("RGB")
        mask_pil = item['mask'].convert("L")

        # 如果 Mask 尺寸和原图不一致，强制缩放 Mask 匹配原图！
        if img_pil.size != mask_pil.size:
            # 使用最近邻插值（NEAREST），防止 Mask 边缘变模糊
            mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.NEAREST)

        # 2. 转换为 Numpy 数组
        img = np.array(img_pil)
        mask = np.array(mask_pil) / 255.0

        # 3. Mask 与图像融合 (抠除背景)
        masked_img = Image.fromarray((img * mask[..., None]).astype(np.uint8))

        if self.transform:
            masked_img = self.transform(masked_img)

        return masked_img, {
            'flower': torch.tensor(item['flower type'], dtype=torch.long),
            'handle': torch.tensor(item['handle type'], dtype=torch.long)
        }