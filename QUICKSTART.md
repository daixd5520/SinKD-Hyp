# MoG-SKD 快速入门指南

## 🎯 什么是MoG-SKD？

**MoG-SKD** = **Mixture-of-Geometries Sinkhorn Knowledge Distillation**

一个用于知识蒸馏的**自适应多几何框架**，根据样本难度自动选择最合适的几何空间。

### 核心创新

```
简单样本 → Euclidean / Fisher-Rao (标准几何)
困难样本 → Hyperbolic (双曲几何，处理层次结构和不确定性)
```

---

## 📦 安装和使用

### 1. 运行测试验证安装

```bash
python test_mog_skd.py
```

**预期输出**：
```
==================================================================
MoG-SKD Test Suite
==================================================================

Testing Fisher-Rao Expert...
  ✓ Fisher-Rao Expert test passed!

Testing Euclidean Expert...
  ✓ Euclidean Expert test passed!

Testing Hyperbolic Expert...
  ✓ Hyperbolic Expert test passed!

Testing Statistical Gating Network...
  ✓ Statistical Gating test passed!

Testing MoGSKD Unified Framework...
  ✓ MoGSKD Framework test passed!

...
==================================================================
All tests passed! ✓
==================================================================

MoG-SKD is ready for KDD submission! 🚀
```

---

### 2. 快速训练示例

```bash
# 基础训练（调试模式）
python train_mog_skd.py \
    --dataset_name "copa" \
    --template_name "justify_this" \
    --model_name_or_path "google/t5-small" \
    --teacher_model_path "google/t5-base" \
    --output_dir "./experiments/quick_test" \
    --use_mog_skd \
    --lambda_reg 0.1 \
    --debug \
    --num_train_epochs 1
```

---

### 3. 完整训练（生产级）

```bash
python train_mog_skd.py \
    --dataset_name "copa" \
    --template_name "justify_this" \
    --model_name_or_path "/path/to/t5-small" \
    --teacher_model_path "/path/to/t5-base-finetuned" \
    --output_dir "./experiments/mog_skd_full" \
    --use_mog_skd \
    --lambda_reg 0.1 \
    --temperature 2.0 \
    --learnable_curvature \
    --hyperbolic_c 1.0 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100
```

---

### 4. 生成论文图表

训练完成后，生成KDD论文所需的"Money Plot"：

```bash
python visualize_mog_skd.py \
    --logs_path "./experiments/mog_skd_full/mog_skd_logs.json" \
    --output_dir "./paper_figures"
```

**生成的图表**：
- `money_plot.png` - ⭐ 专家权重 vs 不确定性（核心图表）
- `expert_losses.png` - 各专家损失曲线
- `gating_entropy.png` - 门控特化过程
- `hyperbolic_curvature.png` - 学习的曲率轨迹

---

## 🔧 代码示例

### 基础使用

```python
from mog_skd import MoGSKD

# 创建模型
mog_skd = MoGSKD(
    T=2.0,              # 温度
    lambda_reg=0.1,     # 门控熵正则化系数
    hidden_dim=32,      # 门控网络隐藏维度
    learnable_curvature=True  # 可学习双曲曲率
)

# 训练循环
for batch in dataloader:
    student_logits = student_model(batch)
    with torch.no_grad():
        teacher_logits = teacher_model(batch)

    # 计算MoG-SKD损失
    loss, logs = mog_skd(
        student_logits,
        teacher_logits,
        return_details=True  # 获取详细日志
    )

    # 反向传播
    loss.backward()
    optimizer.step()

    # 打印统计信息
    if step % 100 == 0:
        print(f"Loss: {loss.item():.4f}")
        print(f"  Fisher: {logs['weight_fisher']:.2f}")
        print(f"  Euclid: {logs['weight_euclid']:.2f}")
        print(f"  Hyper:  {logs['weight_hyper']:.2f}")
```

---

### 高级：使用配置类

```python
from mog_skd import MoGSKDConfig

# 定义配置
config = MoGSKDConfig(
    T=2.0,
    lambda_reg=0.1,
    hidden_dim=32,
    use_sinkhorn=False,      # Euclidean专家是否用Sinkhorn
    learnable_curvature=True,
    hyperbolic_c=1.0
)

# 创建模型
mog_skd = config.create_model()

# 保存配置（用于实验重现）
import json
with open('config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)
```

---

### 只使用单个专家（消融实验）

```python
from losses.experts import FisherRaoExpert, HyperbolicExpert

# 创建单个专家
expert = HyperbolicExpert(T=2.0, c=1.0)

# 计算损失
loss = expert(student_logits, teacher_logits)  # [batch_size]

# 平均损失
total_loss = loss.mean()
total_loss.backward()
```

---

## 📊 理解输出日志

### `logs` 字典内容

```python
{
    # 各专家损失
    'loss_fisher': 0.1234,
    'loss_euclid': 0.2345,
    'loss_hyper': 0.3456,

    # 门控权重（最重要的指标！）
    'weight_fisher': 0.25,
    'weight_euclid': 0.50,
    'weight_hyper': 0.25,

    # 门控统计
    'gating_entropy': 0.98,  # 越低越特化

    # 损失分解
    'distill_loss': 0.25,
    'reg_loss': 0.098,

    # 双曲曲率（如果可学习）
    'hyperbolic_curvature': 1.05,

    # 每样本数据（用于深度分析）
    'per_sample_data': {
        'fisher_loss': tensor([...]),  # [batch_size]
        'euclid_loss': tensor([...]),
        'hyper_loss': tensor([...]),
        'fisher_weight': tensor([...]),
        'euclid_weight': tensor([...]),
        'hyper_weight': tensor([...]),
        'teacher_entropy': tensor([...])  # 不确定性
    }
}
```

---

## 🎨 "Money Plot" 解读

Money Plot（expert_weights_vs_entropy.png）是KDD论文的核心图表。

**X轴**：教师预测熵（不确定性）
- 0 = 完全确定（简单样本）
- 1 = 完全不确定（困难样本）

**Y轴**：专家权重（0-1）
- 展示不同专家在不同难度下的权重

**预期趋势**：
```
高权重（接近1）
    │
    │    Hyperbolic (红)
    │   ╱
    │  ╱  ← 困难样本用双曲几何
    │ ╱
    │╱───── Euclidean (蓝)
    │       ╲
    │        ╲ ← 简单样本用欧氏/Fisher-Rao
    │         ╲───── Fisher-Rao (绿)
    │
    └───────────────────────
    0          熵          1
              (不确定性)
```

**论文写法**：
> "MoG-SKD automatically activates hyperbolic geometry for uncertain samples (high entropy), while simpler geometries handle easy cases. This adaptive selection is achieved through our interpretable statistical gating network."

---

## ⚙️ 超参数调优指南

### 1. `lambda_reg` (门控熵正则化)

**范围**: 0.01 - 0.5

- **0.01**: 弱正则化 → 专家权重更均匀
- **0.1**: 中等正则化（推荐）→ 适度特化
- **0.5**: 强正则化 → 强制选择单一专家

**调优策略**：
```python
# 从0.1开始
# 如果门控坍塌到单一专家 → 减小lambda_reg
# 如果门控过于均匀 → 增大lambda_reg
```

### 2. `temperature` (蒸馏温度)

**范围**: 1.0 - 8.0

- **1.0**: 原始logits（不软化）
- **2.0**: 轻微软化（推荐起点）
- **4.0+**: 强软化， smoother梯度

### 3. `hyperbolic_c` (初始曲率)

**范围**: 0.5 - 2.0

- **< 1.0**: 更平坦的双曲空间
- **= 1.0**: 标准Poincaré盘（推荐）
- **> 1.0**: 更弯曲的双曲空间

如果`learnable_curvature=True`，这只是一个初始值。

### 4. `hidden_dim` (门控网络容量)

**范围**: 16 - 64

- **16**: 小模型，快速训练
- **32**: 标准（推荐）
- **64**: 大模型，更强表达能力

---

## 🐛 常见问题

### Q1: 训练开始时loss很高或NaN？

**解决方案**：
```python
# 1. 降低学习率
--learning_rate 5e-5  # 从1e-4降低

# 2. 检查logits范围
# 在训练循环中添加：
print(f"Student logits range: [{student_logits.min():.2f}, {student_logits.max():.2f}]")

# 3. 减小初始曲率
--hyperbolic_c 0.5  # 从1.0降低
```

### Q2: 门控总是选择同一个专家？

**可能原因**：
- `lambda_reg`太大 → 减小到0.01
- 门控网络容量不足 → 增大`hidden_dim`
- 某个专家损失总是最小 → 检查专家实现

**调试代码**：
```python
# 在训练循环中打印
print(f"Fisher loss: {logs['loss_fisher']:.4f}")
print(f"Euclid loss: {logs['loss_euclid']:.4f}")
print(f"Hyper loss:  {logs['loss_hyper']:.4f}")
```

### Q3: 双曲专家梯度爆炸？

**解决方案**：
```python
# 1. 检查clamp设置
# 在 HyperbolicExpert._exp_map() 中：
v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(1e-6)

# 2. 降低温度
--temperature 1.0  # 从2.0降低

# 3. 使用固定曲率
# 不使用 --learnable_curvature
```

---

## 📚 下一步

1. **运行测试**: `python test_mog_skd.py`
2. **快速实验**: `--debug` 模式快速验证
3. **完整训练**: 10+ epochs，生成完整日志
4. **可视化**: 生成Money Plot和其他图表
5. **撰写论文**: 参考MOG_SKD_README.md中的KDD指南

---

## 🏆 KDD投稿检查清单

- [ ] 运行完整训练（10+ epochs）
- [ ] 生成所有图表（Money Plot, 训练动态, 消融）
- [ ] 填写消融实验表
- [ ] 确认数学严谨性（引用正确）
- [ ] 验证可重现性（随机种子，配置文件）
- [ ] 准备补充材料（代码，数据）

**准备投稿！** 🚀
