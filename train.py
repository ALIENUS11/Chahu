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
    print("=" * 50, flush=True)
    print(" 初始化训练环境...", flush=True)
    print("=" * 50, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f" 当前计算设备: GPU ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        print(" 未检测到 GPU，将使用 CPU 进行训练", flush=True)

    print("\n 正在获取并处理数据集... ", flush=True)
    train_hf, val_hf, test_hf = prepare_data('EN')

    # 动态从数据集中读取实际的类别数量
    num_flower_classes = train_hf.features['flower type'].num_classes
    num_handle_classes = train_hf.features['handle type'].num_classes

    print(f" 数据集真实类别统计:")
    print(f"    花型 (Flower): {num_flower_classes} 类")
    print(f"    提手 (Handle): {num_handle_classes} 类")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("\n 正在封装 DataLoader...", flush=True)
    # 注意：在 Windows 下坚决不要加 num_workers 参数
    train_loader = DataLoader(ChaHuDataset(train_hf, tf), batch_size=32, shuffle=True)
    val_loader = DataLoader(ChaHuDataset(val_hf, tf), batch_size=32)

    print("\n 正在构建 MultiTaskNet 模型与优化器...", flush=True)
    model = MultiTaskNet(num_flower_classes, num_handle_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    print(" 模型初始化完成。", flush=True)

    EPOCHS = 10
    print("\n" + "=" * 50, flush=True)
    print(f" 开始模型训练 (共计 {EPOCHS} Epochs)", flush=True)
    print("=" * 50, flush=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=False)

        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            pf, ph = model(imgs)

            loss = criterion(pf, labels['flower'].to(device)) + criterion(ph, labels['handle'].to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)
        print(f"\n Epoch [{epoch + 1}/{EPOCHS}] 训练完成 | 训练集平均 Loss: {avg_train_loss:.4f}", flush=True)

        print(f" 正在验证集上评估泛化能力...", flush=True)
        run_eval(model, val_loader, device)

        save_path = f"chahu_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f" 模型已保存至: {save_path}", flush=True)
        print("-" * 50, flush=True)

    print("\n 全部训练结束！", flush=True)
    print(" 正在使用测试集进行最终评估...", flush=True)
    test_loader = DataLoader(ChaHuDataset(test_hf, tf), batch_size=32)
    model.load_state_dict(torch.load(f"chahu_model_epoch_{EPOCHS}.pth"))
    run_eval(model, test_loader, device)

if __name__ == "__main__":
    main()