# ==========================================
# 🚨 强行接管底层 DLL 加载顺序 (极其重要，防止 0xC0000005)
# ==========================================
import numpy as np
import datasets
import sklearn
from PIL import Image
import torch
import torchvision

# 然后再导入其他功能和本地模块
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import MultiTaskNet
from utils import prepare_data, ChaHuDataset
from evaluation import run_eval

def main():
    # 🚨 加入 flush=True，强制立刻输出到屏幕，防止崩溃时吞掉日志
    print("=" * 50, flush=True)
    print("🚀 初始化训练环境...", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()