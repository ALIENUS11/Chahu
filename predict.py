import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import MultiTaskNet
from utils import prepare_data


def predict_single_image(img_path, mask_path, weight_path):
    print("=" * 50)
    print("初始化推理环境...")
    print("=" * 50)

    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 自动获取类别映射表 (保证与训练时 100% 一致)
    print("正在加载类别映射表...")
    # 只取 train_hf 获取特征字典即可，耗时很短
    train_hf, _, _ = prepare_data('EN')
    flower_features = train_hf.features['flower type']
    handle_features = train_hf.features['handle type']

    num_flowers = flower_features.num_classes  # 应该是 17
    num_handles = handle_features.num_classes  # 应该是 6

    flower_names = flower_features.names
    handle_names = handle_features.names

    # 3. 初始化模型并加载权重
    print(f"正在加载模型权重: {weight_path}")
    model = MultiTaskNet(num_flower_classes=num_flowers, num_handle_classes=num_handles).to(device)

    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()  # 必须开启评估模式
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 4. 图像读取与防御性预处理
    try:
        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")

        if img_pil.size != mask_pil.size:
            mask_pil = mask_pil.resize(img_pil.size, Image.Resampling.NEAREST)

        img_np = np.array(img_pil)
        mask_np = np.array(mask_pil) / 255.0

        # 背景抠除
        masked_img_np = (img_np * mask_np[..., None]).astype(np.uint8)
        masked_img = Image.fromarray(masked_img_np)

    except Exception as e:
        print(f"图像处理失败，请检查路径或图片格式: {e}")
        return

    # 5. 标准化与张量化
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = tf(masked_img).unsqueeze(0).to(device)

    # 6. 执行推理
    print("\n正在进行前向推理...")
    with torch.no_grad():
        pf_logits, ph_logits = model(input_tensor)

        # 计算 Softmax 概率
        prob_f = torch.softmax(pf_logits, dim=1)
        prob_h = torch.softmax(ph_logits, dim=1)

        # 提取概率最大值的索引和置信度
        f_idx = prob_f.argmax(1).item()
        f_conf = prob_f.max(1)[0].item()

        h_idx = prob_h.argmax(1).item()
        h_conf = prob_h.max(1)[0].item()

    # 7. 打印最终结果
    print("\n" + "-" * 40)
    print("推理结果报告")
    print("-" * 40)
    print(f"目标图片: {img_path}")
    print(f"预测花型: {flower_names[f_idx]} (置信度: {f_conf:.2%})")
    print(f"预测把手: {handle_names[h_idx]} (置信度: {h_conf:.2%})")
    print("-" * 40)


if __name__ == "__main__":
    # 修改为你要测试的图片路径
    # 将 weight_path 改为你训练出来的最终 epoch 权重名，例如 chahu_model_epoch_10.pth
    predict_single_image(
        img_path="test_image.jpg",
        mask_path="test_mask.png",
        weight_path="chahu_model_epoch_10.pth"
    )