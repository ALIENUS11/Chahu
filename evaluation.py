import torch
from sklearn.metrics import classification_report


def run_eval(model, loader, device):
    """
    评估模型：统计两个分类任务的详细指标。
    """
    model.eval()
    f_true, f_pred, h_true, h_pred = [], [], [], []

    with torch.no_grad():  # 推理阶段不计算梯度，节省显存
        for imgs, targets in loader:
            imgs = imgs.to(device)
            out_f, out_h = model(imgs)

            # 取出概率最大的类别索引
            f_pred.extend(out_f.argmax(1).cpu().tolist())
            f_true.extend(targets['flower'].tolist())
            h_pred.extend(out_h.argmax(1).cpu().tolist())
            h_true.extend(targets['handle'].tolist())

    # classification_report 会输出每个类别的 Precision, Recall, F1
    # 对于不平衡数据，主要看 'macro avg' (宏平均)，而非 'accuracy'
    print("\n[Flower Type 分类报告]")
    print(classification_report(f_true, f_pred, zero_division=0))
    print("\n[Handle Type 分类报告]")
    print(classification_report(h_true, h_pred, zero_division=0))