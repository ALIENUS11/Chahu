from collections import Counter
from datasets import load_dataset, DatasetDict

# 1. 加载 EN 数据集
print("正在加载 EN 数据集...")
en_dataset = load_dataset("AGI-FBHC/ChaHu", split='EN')
initial_total = len(en_dataset)

# 2. 统计并合并样本数少 10 的类别
# 这是为了防止 train_test_split 在分层采样时因样本太少而崩溃
flower_counts = Counter(en_dataset['flower type'])
rare_classes = {k for k, v in flower_counts.items() if v < 10 and k not in ['Else', 'Other']}

if rare_classes:
    print(f"检测到罕见类别，正在合并至 'Else': {rare_classes}")
    def merge_func(example):
        if example['flower type'] in rare_classes:
            example['flower type'] = 'Else'
        return example
    en_dataset = en_dataset.map(merge_func)

# 3. 类别编码
en_dataset = en_dataset.class_encode_column("flower type")

# 4. 执行 8:1:1 划分
# 第一次：切出 20% 作为临时集 (Val + Test)
train_temp_split = en_dataset.train_test_split(
    test_size=0.2,
    stratify_by_column="flower type",
    seed=42
)

# 第二次：将临时集平分为 10% 验证集和 10% 测试集
val_test_split = train_temp_split['test'].train_test_split(
    test_size=0.5,
    stratify_by_column="flower type",
    seed=42
)

# 5. 组合最终结果
en_final_datasets = DatasetDict({
    'train': train_temp_split['train'],
    'validation': val_test_split['train'],
    'test': val_test_split['test']
})

# ==========================================
# 最终长度输出
# ==========================================
print("\n" + "="*30)
print("📊 EN 数据集划分长度统计")
print("="*30)
print(f"总原始数据量: {initial_total}")
print(f"训练集 (Train): {len(en_final_datasets['train'])}")
print(f"验证集 (Val)  : {len(en_final_datasets['validation'])}")
print(f"测试集 (Test) : {len(en_final_datasets['test'])}")
print("="*30)