# 双曲空间知识蒸馏改进方案

## 📋 项目背景

本项目实现了基于Sinkhorn距离的知识蒸馏（Knowledge Distillation），并在双曲空间（Lorentz模型）中计算概率分布之间的距离。相关论文发表在COLING 2024和TNNLS 2024。

**核心目标**：通过最小化教师模型和学生模型输出概率分布之间的Sinkhorn距离，实现知识蒸馏。

## 🔍 当前实现详解

### 1. 整体流程

```python
# 文件位置: loss.py, Sinkhorn类 (lines 119-210)

class Sinkhorn(nn.Module):
    def __init__(self, T, use_hyperbolic=True, curvature=1.0, learnable_curvature=False):
        self.use_hyperbolic = use_hyperbolic  # 是否使用双曲空间
        self.curvature = curvature            # 双曲空间曲率

    def forward(self, y_s, y_t, mode="classification"):
        # 步骤1: Logits -> 概率分布
        p_s = softmax(y_s / self.T)  # 学生模型概率
        p_t = softmax(y_t / self.T)  # 教师模型概率

        # 步骤2: 计算Sinkhorn损失
        emd_loss = 0.001 * self.sinkhorn_loss(x=p_s, y=p_t)
        return emd_loss
```

### 2. 双曲空间映射（核心部分）

#### 2.1 Lorentz指数映射

```python
def _lorentz_expmap0(self, x, eps=1e-5):
    """
    将欧几里得空间的概率向量映射到双曲面

    数学原理:
    - 基点: o = (1/√c, 0, ..., 0)
    - 切向量: v ∈ T_oH (欧几里得空间)
    - 映射公式:
        exp_o(v) = (cosh(√c·||v||)/√c, sinh(√c·||v||)/(√c·||v||) · v)

    输入: x - 概率向量 [batch_size, num_classes]
    输出: 双曲面上的点 [batch_size, num_classes + 1]
    """
    curvature = self._get_curvature()  # 曲率 c
    sqrt_c = torch.sqrt(curvature)
    norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scaled_norm = sqrt_c * norm

    # 时间分量（第一维）
    time = torch.cosh(scaled_norm) / sqrt_c

    # 空间分量（其余维度）
    space = torch.sinh(scaled_norm) * x / (scaled_norm * sqrt_c)

    # 投影到双曲面，满足 ⟨x,x⟩_L = -1/c
    return self._lorentz_project(torch.cat([time, space], dim=-1))
```

#### 2.2 双曲距离计算

```python
def _hyperbolic_distance(self, x, y, eps=1e-5):
    """
    计算双曲面上的测地线距离

    数学公式:
    d_L(x, y) = (1/√c) · acosh(-c · ⟨x, y⟩_L)

    其中 Lorentz 内积定义为:
    ⟨x, y⟩_L = -x_0·y_0 + x_1·y_1 + ... + x_n·y_n
    """
    curvature = self._get_curvature()
    x_expanded = self._lorentz_project(x).unsqueeze(1)
    y_expanded = self._lorentz_project(y).unsqueeze(0)

    # Minkowski内积
    minkowski = self._lorentz_inner(x_expanded, y_expanded)

    # 计算距离
    argument = (-curvature * minkowski).clamp_min(1 + eps)
    return (torch.acosh(argument) / torch.sqrt(curvature)).squeeze(-1)
```

#### 2.3 Sinkhorn损失计算

```python
def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=20):
    """
    使用Sinkhorn算法计算近似Earth Mover's Distance

    步骤:
    1. 将概率分布映射到双曲空间
    2. 计算双曲距离矩阵
    3. 使用Sinkhorn迭代求解最优传输计划
    4. 计算传输成本
    """
    if self.use_hyperbolic:
        # 映射到双曲空间
        x_lorentz = self._lorentz_expmap0(x)
        y_lorentz = self._lorentz_expmap0(y)

        # 计算双曲距离矩阵
        Wxy = self._hyperbolic_distance(x_lorentz, y_lorentz)
    else:
        # 欧几里得距离（L1）
        Wxy = torch.cdist(x, y, p=1)

    # Sinkhorn算法
    K = torch.exp(-Wxy / epsilon)
    P = self.sinkhorn_normalized(K, n_iters)

    return torch.sum(P * Wxy)
```

### 3. 使用示例

```python
# 训练脚本: main_glue_distill.py

# 初始化（默认使用双曲空间）
saliency_function = Sinkhorn(T=1).to(device)

# 前向传播
loss = saliency_function(student_logits, teacher_logits)
```

## ⚠️ 当前实现的核心问题

### 问题1: 概率分布与双曲空间的几何不匹配

**问题描述**：
- Softmax输出的是**概率单纯形**（Simplex）上的点: {p | ∑p_i = 1, p_i ≥ 0}
- 概率分布的自然几何是**信息几何**（Fisher信息度量），而非双曲几何
- 双曲空间适用于**层次化数据**（如树形结构），但分类任务的类别不一定有层次关系

**具体影响**：
```python
# 示例: 两个概率分布
p1 = [0.9, 0.05, 0.05]  # 高度集中在第1类
p2 = [0.85, 0.10, 0.05]  # 略有差异

# 在双曲空间中:
# - 它们被映射到双曲面某处
# - 但这个映射的概率含义不清晰
# - 为什么概率的"差异"应该在双曲空间度量？
```

**理论缺陷**：
- 缺少理论证明：为什么概率分布的双曲嵌入能更好地表示"知识"？
- 未考虑概率分布的约束条件（∑p_i = 1）

### 问题2: Sinkhorn算法在双曲空间中的理论依据不足

**问题描述**：
- Sinkhorn算法基于**最优传输理论**（Optimal Transport）
- 传统OT理论在欧几里得空间中定义
- 直接套用到双曲空间缺少理论推导

**关键疑问**：
1. 双曲空间中的Wasserstein距离如何正确定义？
2. Sinkhorn迭代在双曲几何下是否收敛？
3. 当前的实现是否真的是"双曲空间的最优传输"？

**当前代码的做法**：
```python
# 直接将欧几里得空间的算法套用
Wxy = self._hyperbolic_distance(x_lorentz, y_lorentz)  # 双曲距离
K = torch.exp(-Wxy / epsilon)  # 但这个exp在双曲空间中是否合理？
P = self.sinkhorn_normalized(K, n_iters)  # 归一化在双曲空间中如何定义？
```

### 问题3: 缺少切空间映射

**数学问题**：
- Lorentz指数映射 `exp_o: T_oH → H` 需要输入**切向量**
- 当前实现直接将概率向量作为切向量
- 但概率向量不一定在切空间 T_oH 中

**正确做法应该是**：
```python
# 步骤1: 将概率分布投影到切空间
# 步骤2: 在切空间中应用指数映射
```

### 问题4: 数值稳定性问题

**代码位置**: loss.py:186
```python
argument = (-curvature * minkowski).clamp_min(1 + eps)
return (torch.acosh(argument) / torch.sqrt(curvature)).squeeze(-1)
```

**问题**：
- 当两个分布非常接近时，`minkowski` 接近 `-1/c`
- `acosh(argument)` 接近0，梯度不稳定
- 虽然有 `eps=1e-5` 保护，但训练中仍可能数值溢出

### 问题5: 缺少消融实验验证

**需要验证的假设**：
1. 双曲空间是否真的比欧几里得空间好？
2. 曲率参数的敏感性如何？
3. 哪些任务/数据集适合双曲空间？

## 🎯 改进目标

### 目标1: 理论严谨性
- [ ] 选择更适合概率分布的几何空间
- [ ] 提供清晰的数学推导和理论依据
- [ ] 确保所有操作在数学上是合理的

### 目标2: 实现正确性
- [ ] 修复切空间映射问题
- [ ] 改进数值稳定性
- [ ] 添加理论验证（如流形约束检查）

### 目标3: 实验验证
- [ ] 实现消融实验（欧几里得 vs 双曲 vs 信息几何）
- [ ] 在多个数据集上验证有效性
- [ ] 分析曲率参数的影响

### 目标4: 可扩展性
- [ ] 支持多种距离度量（双曲、欧几里得、Wasserstein、Fisher-Rao）
- [ ] 支持可学习的几何参数
- [ ] 易于添加新的几何空间

## 💡 改进方案选项

### 方案A: 信息几何方法（推荐⭐⭐⭐⭐⭐）

**理论依据**：概率分布的自然几何是Fisher信息度量

**实现思路**：
```python
class FisherRaoDistance(nn.Module):
    """
    使用Fisher-Rao距离度量概率分布差异

    数学原理:
    - 概率单纯形配备Fisher信息度量
    - Fisher-Rao距离是信息几何中的测地线距离
    - 对于多项式分布，可通过计算求得
    """

    def forward(self, p, q):
        # 方法1: 使用sqrt变换
        # Fisher-Rao距离等价于L2距离（经过sqrt变换）
        sqrt_p = torch.sqrt(p + 1e-8)
        sqrt_q = torch.sqrt(q + 1e-8)
        fr_distance = torch.norm(sqrt_p - sqrt_q, p=2, dim=-1)
        return fr_distance

    def sinkhorn_loss_fisher(self, x, y, epsilon=0.1, n_iters=20):
        # 在Fisher-Rao几何中计算Sinkhorn
        Wxy = self._fisher_rao_distance_matrix(x, y)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)

    def _fisher_rao_distance_matrix(self, x, y):
        """计算批量Fisher-Rao距离矩阵"""
        sqrt_x = torch.sqrt(x + 1e-8).unsqueeze(1)
        sqrt_y = torch.sqrt(y + 1e-8).unsqueeze(0)
        return torch.norm(sqrt_x - sqrt_y, p=2, dim=-1)
```

**优点**：
- ✅ 理论基础扎实（信息几何）
- ✅ 天然适合概率分布
- ✅ 计算高效（类似欧几里得距离）
- ✅ 数值稳定

**实现要求**：
1. 实现Fisher-Rao距离矩阵计算
2. 在Fisher-Rao几何中重新推导Sinkhorn算法
3. 添加理论验证（如对称性、三角不等式）

---

### 方案B: 改进的双曲空间方法

**理论依据**：为双曲空间映射提供更好的理论基础

**改进点1: 重新设计映射方式**

```python
class ImprovedHyperbolicSinkhorn(nn.Module):
    """
    改进的双曲空间Sinkhorn距离
    """

    def __init__(self, T, curvature=1.0, use_temperature=True):
        super().__init__()
        self.T = T
        self.curvature = curvature
        self.use_temperature = use_temperature

    def _softmax_to_simplex(self, logits):
        """标准softmax，输出在单纯形上"""
        return F.softmax(logits / self.T, dim=-1)

    def _simplex_to_tangent(self, p):
        """
        将概率分布映射到双曲面的切空间

        关键改进: 不是直接使用概率向量，而是使用对数几率
        原因: log odds 在黎曼流形中有明确的几何意义
        """
        # 使用log odds作为切向量
        # logit = log(p / (1-p)) 对于二分类
        # 对于多分类: log(p_i / p_reference)
        eps = 1e-8
        log_p = torch.log(p + eps)
        # 中心化（相对于均匀分布）
        uniform = torch.ones_like(p) / p.size(-1)
        log_uniform = torch.log(uniform + eps)
        tangent = log_p - log_uniform
        return tangent

    def _lorentz_expmap0_improved(self, v, eps=1e-5):
        """
        改进的指数映射，确保输入在切空间中

        v: 切向量 [batch_size, num_classes]
        """
        curvature = self.curvature
        sqrt_c = torch.sqrt(curvature)

        # 计算切向量的范数
        norm_v = torch.norm(v, dim=-1, keepdim=True).clamp_min(eps)

        # 双曲指数映射
        scaled_norm = sqrt_c * norm_v

        # 时间分量
        time = torch.cosh(scaled_norm) / sqrt_c

        # 空间分量 - 注意这里v是切向量
        space = torch.sinh(scaled_norm) * v / (scaled_norm * sqrt_c)

        # 组合并投影到双曲面
        point = torch.cat([time, space], dim=-1)
        return self._lorentz_project(point)

    def forward(self, y_s, y_t):
        # 步骤1: Logits -> 概率分布
        p_s = self._softmax_to_simplex(y_s)
        p_t = self._softmax_to_simplex(y_t)

        # 步骤2: 概率 -> 切空间
        v_s = self._simplex_to_tangent(p_s)
        v_t = self._simplex_to_tangent(p_t)

        # 步骤3: 切空间 -> 双曲空间
        h_s = self._lorentz_expmap0_improved(v_s)
        h_t = self._lorentz_expmap0_improved(v_t)

        # 步骤4: 计算双曲距离矩阵
        Wxy = self._hyperbolic_distance(h_s, h_t)

        # 步骤5: Sinkhorn算法
        return self._sinkhorn(Wxy)
```

**改进点2: 添加流形约束检查**

```python
def _check_manifold_constraint(self, x):
    """
    检查点是否在双曲面上

    约束: ⟨x, x⟩_L = -1/c
    """
    curvature = self.curvature
    inner_prod = self._lorentz_inner(x, x)
    expected = -1.0 / curvature

    # 允许数值误差
    tolerance = 1e-4
    assert torch.allclose(inner_prod, expected, atol=tolerance), \
        f"Manifold constraint violated: {inner_prod} != {expected}"

def _lorentz_project(self, x, eps=1e-8):
    """
    投影到双曲面，满足 ⟨x, x⟩_L = -1/c

    改进: 添加详细的数值稳定性处理
    """
    curvature = self.curvature
    space = x[..., 1:]

    # 改进的数值稳定性
    sum_space_sq = torch.sum(space * space, dim=-1, keepdim=True)
    time_squared = (sum_space_sq + 1 / curvature).clamp_min(eps)
    time = torch.sqrt(time_squared)

    projected = torch.cat([time, space], dim=-1)

    # 验证投影结果
    if self.training:
        self._check_manifold_constraint(projected)

    return projected
```

**改进点3: 数值稳定的双曲距离**

```python
def _hyperbolic_distance_stable(self, x, y, eps=1e-5):
    """
    数值稳定的双曲距离计算

    改进: 使用双曲正弦和余弦的数值稳定版本
    """
    curvature = self.curvature
    x = self._lorentz_project(x)
    y = self._lorentz_project(y)

    x_expanded = x.unsqueeze(1)
    y_expanded = y.unsqueeze(0)

    # Minkowski内积
    minkowski = self._lorentz_inner(x_expanded, y_expanded)

    # 数值稳定的acosh计算
    # acosh(x) = log(x + sqrt(x^2 - 1))
    argument = (-curvature * minkowski).clamp_min(1 + eps)

    # 方法1: 使用torch.acosh（可能不稳定）
    # distance = torch.acosh(argument) / torch.sqrt(curvature)

    # 方法2: 使用log形式（更稳定）
    sqrt_term = torch.sqrt(argument * argument - 1 + eps)
    distance = torch.log(argument + sqrt_term) / torch.sqrt(curvature)

    return distance.squeeze(-1)
```

**优点**：
- ✅ 改进了切空间映射的理论基础
- ✅ 添加了流形约束验证
- ✅ 提高了数值稳定性
- ✅ 保留了双曲空间的特性

**需要实现**：
1. `_simplex_to_tangent`: 概率到切空间的映射
2. `_check_manifold_constraint`: 流形约束检查
3. `_hyperbolic_distance_stable`: 稳定的距离计算
4. 消融实验：log odds vs 原始概率

---

### 方案C: 混合方法（自适应选择几何）

**核心思想**：根据数据特性自适应选择最合适的几何空间

```python
class AdaptiveGeometricSinkhorn(nn.Module):
    """
    自适应几何空间的Sinkhorn距离

    支持:
    1. 欧几里得几何 (Euclidean)
    2. 双曲几何 (Hyperbolic)
    3. Fisher-Rao几何 (信息几何)
    4. 混合几何 (学习权重组合)
    """

    def __init__(self, T, geometries=['euclidean', 'fisher'], learnable_weights=True):
        super().__init__()
        self.T = T
        self.geometries = geometries

        # 可学习的几何权重
        if learnable_weights:
            self.geometry_weights = nn.Parameter(torch.ones(len(geometries)) / len(geometries))
        else:
            self.register_buffer('geometry_weights', torch.ones(len(geometries)) / len(geometries))

        # 初始化各个几何模块
        self.euclidean_sinkhorn = EuclideanSinkhorn(T)
        self.fisher_sinkhorn = FisherRaoSinkhorn(T)
        self.hyperbolic_sinkhorn = HyperbolicSinkhorn(T)

    def forward(self, y_s, y_t):
        # 计算各个几何空间的损失
        losses = []
        if 'euclidean' in self.geometries:
            losses.append(self.euclidean_sinkhorn(y_s, y_t))
        if 'fisher' in self.geometries:
            losses.append(self.fisher_sinkhorn(y_s, y_t))
        if 'hyperbolic' in self.geometries:
            losses.append(self.hyperbolic_sinkhorn(y_s, y_t))

        # 加权组合
        weights = F.softmax(self.geometry_weights, dim=0)
        total_loss = sum(w * loss for w, loss in zip(weights, losses))

        return total_loss

    def get_geometry_importance(self):
        """返回各个几何空间的重要性"""
        return F.softmax(self.geometry_weights, dim=0).detach().cpu().numpy()
```

**使用示例**：
```python
# 初始化混合几何
sinkhorn = AdaptiveGeometricSinkhorn(
    T=2,
    geometries=['euclidean', 'fisher', 'hyperbolic'],
    learnable_weights=True
)

# 训练
for epoch in range(num_epochs):
    loss = sinkhorn(student_logits, teacher_logits)
    loss.backward()
    optimizer.step()

    # 打印几何空间的重要性
    importance = sinkhorn.get_geometry_importance()
    print(f"Epoch {epoch}: Euclidean={importance[0]:.2f}, "
          f"Fisher={importance[1]:.2f}, Hyperbolic={importance[2]:.2f}")
```

**优点**：
- ✅ 自动选择最合适的几何空间
- ✅ 可以发现哪些几何对特定任务有效
- ✅ 提供可解释性（几何重要性分析）

---

### 方案D: Wasserstein距离方法

**理论依据**：直接在概率单纯形上使用Wasserstein距离

```python
class WassersteinSinkhorn(nn.Module):
    """
    基于Wasserstein距离的Sinkhorn算法

    改进: 使用概率单纯形上的最优传输理论
    """

    def __init__(self, T, ground_metric='euclidean', epsilon=0.1):
        super().__init__()
        self.T = T
        self.ground_metric = ground_metric  # 'euclidean', 'cosine', 'kl'
        self.epsilon = epsilon

    def _cost_matrix(self, p_s, p_t):
        """
        计算成本矩阵（在概率空间中）

        Options:
        1. Euclidean距离（平方）
        2. Cosine距离
        3. KL散度（不对称，需谨慎）
        """
        if self.ground_metric == 'euclidean':
            # L2距离的平方
            p_s_expanded = p_s.unsqueeze(1)
            p_t_expanded = p_t.unsqueeze(0)
            cost = torch.sum((p_s_expanded - p_t_expanded) ** 2, dim=-1)

        elif self.ground_metric == 'cosine':
            # Cosine距离
            p_s_norm = F.normalize(p_s, p=2, dim=-1)
            p_t_norm = F.normalize(p_t, p=2, dim=-1)
            similarity = torch.mm(p_s_norm, p_t_norm.t())
            cost = 1 - similarity

        elif self.ground_metric == 'kl':
            # KL散度（作为成本矩阵，不是对称的）
            # 使用对称版本: KL(P||Q) + KL(Q||P)
            p_s_safe = p_s + 1e-8
            p_t_safe = p_t + 1e-8

            kl_st = torch.sum(p_s_safe * torch.log(p_s_safe / p_t_safe), dim=-1)
            kl_ts = torch.sum(p_t_safe * torch.log(p_t_safe / p_s_safe), dim=-1)

            # 构造成本矩阵
            p_s_expanded = p_s.unsqueeze(1)
            p_t_expanded = p_t.unsqueeze(0)
            kl_st_matrix = kl_st.unsqueeze(1).expand(-1, p_t.size(0))
            kl_ts_matrix = kl_ts.unsqueeze(0).expand(p_s.size(0), -1)

            cost = (kl_st_matrix + kl_ts_matrix) / 2

        return cost

    def forward(self, y_s, y_t, n_iters=20):
        # 步骤1: Softmax
        p_s = F.softmax(y_s / self.T, dim=-1)
        p_t = F.softmax(y_t / self.T, dim=-1)

        # 步骤2: 计算成本矩阵（在概率空间）
        C = self._cost_matrix(p_s, p_t)

        # 步骤3: Sinkhorn算法（标准实现）
        K = torch.exp(-C / self.epsilon)
        P = self._sinkhorn_normalized(K, n_iters)

        # 步骤4: 计算Wasserstein距离
        wasserstein_dist = torch.sum(P * C)

        return wasserstein_dist

    def _sinkhorn_normalized(self, K, n_iters=20):
        """标准的Sinkhorn迭代"""
        P = K.clone()
        for _ in range(n_iters):
            P = P / torch.sum(P, dim=1, keepdim=True)
            P = P / torch.sum(P, dim=0, keepdim=True)
        return P
```

**优点**：
- ✅ 理论基础扎实（经典的最优传输）
- ✅ 在概率空间中操作，几何意义明确
- ✅ 可以选择不同的ground metric
- ✅ 计算效率高

**需要实现**：
1. 多种成本矩阵的计算方法
2. 数值稳定的Sinkhorn算法（对数域计算）
3. 对称性验证（对于对称的成本度量）

---

### 方案E: 理论驱动的双曲空间改进（最理论严谨）

**核心思想**：为双曲空间映射提供严格的理论基础

#### 步骤1: 重新定义概率表示

```python
class TheoreticalHyperbolicKD(nn.Module):
    """
    理论驱动的双曲空间知识蒸馏

    核心改进:
    1. 使用对数几率（log-odds）而非概率
    2. 明确的切空间映射
    3. 理论验证的流形操作
    """

    def __init__(self, T, curvature=1.0, use_poincare=False):
        super().__init__()
        self.T = T
        self.curvature = curvature
        self.use_poincare = use_poincare  # 可选: Poincaré球模型

    def _logits_to_log_odds(self, logits):
        """
        Logits -> Log-odds representation

        理论依据:
        - Log-odds: log(p / (1-p)) for binary
        - Multi-class: log(p_i / p_reference)
        - 这是统计学中的canonical representation
        """
        # 方法1: 相对于参考类的log-odds
        probs = F.softmax(logits / self.T, dim=-1)
        eps = 1e-8
        log_probs = torch.log(probs + eps)

        # 使用最后一类作为参考
        log_odds = log_probs[..., :-1] - log_probs[..., -1:]
        return log_odds

    def _log_odds_to_tangent(self, log_odds):
        """
        Log-odds -> 切空间向量

        理论依据:
        - Log-odds空间是平坦的（可以看作欧几里得空间）
        - 可以直接作为双曲面的切向量
        """
        # 可选: 标准化
        log_odds_norm = F.normalize(log_odds, p=2, dim=-1)
        return log_odds_norm

    def _lorentz_expmap0_rigorous(self, v):
        """
        严格的Lorentz指数映射

        输入: v - 切空间中的向量
        输出: 双曲面上的点
        """
        curvature = self.curvature
        sqrt_c = torch.sqrt(curvature)

        # 切向量的范数
        v_norm = torch.norm(v, dim=-1, keepdim=True)
        v_norm = torch.clamp(v_norm, min=1e-8)

        # 指数映射公式
        sqrt_c_v_norm = sqrt_c * v_norm

        # 时间分量: cosh(√c·||v||) / √c
        time = torch.cosh(sqrt_c_v_norm) / sqrt_c

        # 空间分量: sinh(√c·||v||) / (√c·||v||) · v
        sinh_term = torch.sinh(sqrt_c_v_norm)
        space = sinh_term * v / (sqrt_c_v_norm * sqrt_c)

        # 投影到双曲面
        point = torch.cat([time, space], dim=-1)
        return self._lorentz_project_rigorous(point)

    def _lorentz_project_rigorous(self, x):
        """
        严格的Lorentz投影

        确保点满足: ⟨x, x⟩_L = -1/c
        """
        curvature = self.curvature
        time = x[..., :1]
        space = x[..., 1:]

        # 投影公式: t = sqrt(||space||^2 + 1/c)
        space_norm_sq = torch.sum(space * space, dim=-1, keepdim=True)
        time_new = torch.sqrt(space_norm_sq + 1.0 / curvature)

        # 确保时间分量为正（双曲面的上半叶）
        time_new = torch.abs(time_new)

        projected = torch.cat([time_new, space], dim=-1)

        # 验证
        if self.training:
            self._verify_manifold_constraint(projected)

        return projected

    def _verify_manifold_constraint(self, x):
        """验证流形约束"""
        curvature = self.curvature
        inner = self._lorentz_inner(x, x)
        expected = -1.0 / curvature

        # 计算误差
        error = torch.abs(inner - expected).max().item()

        if error > 1e-3:
            print(f"Warning: Manifold constraint violation, error={error:.6f}")

        assert error < 1e-2, f"Manifold constraint violated: {inner} != {expected}"

    def _lorentz_inner(self, x, y):
        """Lorentz内积: ⟨x, y⟩_L = -x_0·y_0 + Σ x_i·y_i"""
        time_prod = -x[..., :1] * y[..., :1]
        space_prod = torch.sum(x[..., 1:] * y[..., 1:], dim=-1, keepdim=True)
        return time_prod + space_prod

    def _hyperbolic_distance_rigorous(self, x, y):
        """
        严格的双曲距离

        公式: d(x, y) = (1/√c) · acosh(-c · ⟨x, y⟩_L)
        """
        curvature = self.curvature
        sqrt_c = torch.sqrt(curvature)

        # 投影确保在流形上
        x = self._lorentz_project_rigorous(x)
        y = self._lorentz_project_rigorous(y)

        # 扩展维度以计算成对距离
        x_exp = x.unsqueeze(1)  # [batch, 1, dim]
        y_exp = y.unsqueeze(0)  # [1, batch, dim]

        # Lorentz内积
        inner = self._lorentz_inner(x_exp, y_exp)

        # 计算距离
        argument = (-curvature * inner).clamp_min(1.0 + 1e-8)

        # 数值稳定的acosh
        # acosh(z) = log(z + sqrt(z^2 - 1))
        z = argument
        sqrt_term = torch.sqrt(z * z - 1 + 1e-8)
        acosh_result = torch.log(z + sqrt_term)

        distance = acosh_result / sqrt_c
        return distance.squeeze(-1)

    def sinkhorn_loss(self, y_s, y_t, epsilon=0.1, n_iters=20):
        """
        改进的Sinkhorn损失

        关键改进: 在双曲空间中进行所有操作
        """
        # 步骤1: Logits -> Log-odds
        log_odds_s = self._logits_to_log_odds(y_s)
        log_odds_t = self._logits_to_log_odds(y_t)

        # 步骤2: Log-odds -> 切空间
        v_s = self._log_odds_to_tangent(log_odds_s)
        v_t = self._log_odds_to_tangent(log_odds_t)

        # 步骤3: 切空间 -> 双曲空间
        h_s = self._lorentz_expmap0_rigorous(v_s)
        h_t = self._lorentz_expmap0_rigorous(v_t)

        # 步骤4: 计算双曲距离矩阵
        Wxy = self._hyperbolic_distance_rigorous(h_s, h_t)

        # 步骤5: Sinkhorn算法
        # 注意: 这里仍在使用欧几里得空间的Sinkhorn
        # 理论上需要在双曲空间中重新推导
        K = torch.exp(-Wxy / epsilon)
        P = self._sinkhorn_normalized(K, n_iters)

        # 步骤6: 计算损失
        loss = torch.sum(P * Wxy)

        return loss

    def _sinkhorn_normalized(self, K, n_iters=20):
        """Sinkhorn归一化"""
        P = K.clone()
        for _ in range(n_iters):
            P = P / torch.sum(P, dim=1, keepdim=True)
            P = P / torch.sum(P, dim=0, keepdim=True)
        return P

    def forward(self, y_s, y_t):
        return self.sinkhorn_loss(y_s, y_t)
```

**优点**：
- ✅ 最严谨的理论基础
- ✅ 清晰的数学推导
- ✅ 包含验证机制
- ✅ 详细的理论注释

**需要进一步研究**：
1. 双曲空间中的Sinkhorn算法理论
2. Log-odds表示的合理性验证
3. 与概率分布的关系

---

## 📊 实验验证方案

无论选择哪个方案，都需要进行以下实验验证：

### 实验1: 消融实验

**目的**：对比不同几何空间的效果

```python
# 实验设置
geometries = {
    'euclidean': EuclideanSinkhorn(T=2),
    'fisher': FisherRaoSinkhorn(T=2),
    'hyperbolic_original': OriginalHyperbolicSinkhorn(T=2),
    'hyperbolic_improved': ImprovedHyperbolicSinkhorn(T=2),
    'wasserstein': WassersteinSinkhorn(T=2),
}

results = {}
for name, model in geometries.items():
    acc = train_and_evaluate(model, train_set, test_set)
    results[name] = acc

# 可视化
plot_results(results)
```

**评估指标**：
- 准确率（Accuracy）
- 训练稳定性（Loss曲线）
- 收敛速度
- 数值稳定性（NaN/Inf次数）

### 实验2: 敏感性分析

**目的**：分析超参数的影响

```python
# 曲率敏感性（双曲空间）
curvatures = [0.1, 0.5, 1.0, 2.0, 5.0]
for c in curvatures:
    model = HyperbolicSinkhorn(T=2, curvature=c)
    acc = train_and_evaluate(model, ...)
    print(f"Curvature {c}: Accuracy = {acc}")

# Temperature敏感性
temperatures = [1.0, 2.0, 4.0, 8.0]
for T in temperatures:
    model = FisherRaoSinkhorn(T=T)
    acc = train_and_evaluate(model, ...)
    print(f"Temperature {T}: Accuracy = {acc}")

# Epsilon敏感性（Sinkhorn正则化）
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]
for eps in epsilons:
    model = FisherRaoSinkhorn(T=2, epsilon=eps)
    acc = train_and_evaluate(model, ...)
    print(f"Epsilon {eps}: Accuracy = {acc}")
```

### 实验3: 数据集特性分析

**目的**：找出哪些数据集适合双曲空间

```python
datasets = {
    'MNLI': load_mnli(),
    'QQP': load_qqp(),
    'CoLA': load_cola(),
    'SST-2': load_sst2(),
}

# 计算数据集特性
for name, data in datasets.items():
    # 1. 层次性指标
    hierarchy_score = compute_hierarchy_score(data)

    # 2. 概率分布熵
    entropy = compute_average_entropy(data)

    # 3. 类间距离分布
    distance_dist = compute_distance_distribution(data)

    print(f"{name}: Hierarchy={hierarchy_score:.2f}, "
          f"Entropy={entropy:.2f}")

    # 在该数据集上测试不同几何
    for geom_name, geom_model in geometries.items():
        acc = train_and_evaluate(geom_model, data)
        print(f"  {geom_name}: {acc:.4f}")
```

### 实验4: 数值稳定性测试

```python
def test_numerical_stability(model, test_cases):
    """测试模型的数值稳定性"""
    nan_count = 0
    inf_count = 0
    grad_explosion_count = 0

    for y_s, y_t in test_cases:
        try:
            loss = model(y_s, y_t)

            # 检查NaN
            if torch.isnan(loss).any():
                nan_count += 1

            # 检查Inf
            if torch.isinf(loss).any():
                inf_count += 1

            # 检查梯度爆炸
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float('inf')
            )
            if grad_norm > 1000:
                grad_explosion_count += 1

        except Exception as e:
            print(f"Error: {e}")

    print(f"NaN: {nan_count}/{len(test_cases)}")
    print(f"Inf: {inf_count}/{len(test_cases)}")
    print(f"Grad explosion: {grad_explosion_count}/{len(test_cases)}")
```

---

## 🎯 推荐的实施路径

### 阶段1: 快速验证（1-2周）

**目标**：验证哪个几何空间最有效

```python
# 1. 实现Fisher-Rao方法（方案A）
# 2. 对比原始双曲空间
# 3. 在2-3个GLUE任务上快速测试
# 4. 分析结果，决定下一步
```

**代码结构**：
```python
# loss_enhanced.py
class FisherRaoSinkhorn(nn.Module):
    """实现方案A"""
    pass

class ImprovedHyperbolicSinkhorn(nn.Module):
    """实现方案B"""
    pass

# run_ablation.py
def run_ablation_study():
    """运行消融实验"""
    results = {}
    # 对比所有方法
    return results
```

### 阶段2: 深入改进（2-3周）

**目标**：根据阶段1结果深入改进

如果Fisher-Rao最好：
- [ ] 添加更多信息几何理论
- [ ] 优化计算效率
- [ ] 撰写理论说明文档

如果双曲空间最好：
- [ ] 实现方案B或E（改进的双曲空间）
- [ ] 添加理论验证
- [ ] 研究为什么双曲空间有效

如果混合方法最好：
- [ ] 实现方案C
- [ ] 分析几何权重
- [ ] 研究任务特性与几何选择的关系

### 阶段3: 完善与论文（2-4周）

**目标**：完善实现并准备论文/报告

- [ ] 完整的实验报告
- [ ] 可视化工具
- [ ] 理论证明/推导
- [ ] 代码文档和注释

---

## 📚 理论参考资源

### 信息几何

1. **Amari, 2016**: "Information Geometry and Its Applications"
   - 概率分布的Fisher信息度量
   - 流形上的统计推断

2. **Calcetti, 2020**: "An introduction to information geometry"
   - Fisher-Rao距离的计算
   - 与最优传输的关系

### 双曲几何

1. **Nickel & Kiela, 2017**: "Poincaré Embeddings for Learning Hierarchical Representations"
   - 双曲空间在机器学习中的应用
   - 梯度下降在双曲空间中的实现

2. **Bose et al., 2020**: "Latent Variable Modeling with Hyperbolic Normalizing Flows"
   - Lorentz模型的详细实现
   - 数值稳定的双曲操作

### 最优传输

1. **Peyré & Cuturi, 2019**: "Computational Optimal Transport"
   - Sinkhorn算法的理论基础
   - 熵正则化最优传输

2. **Cuturi et al., 2020**: "Sinkhorn Distances"
   - 离散分布的Wasserstein距离
   - 数值稳定算法

---

## 🤔 向AI助手提问的建议提示词

当你向ChatGPT或其他AI助手咨询时，可以使用以下结构化的提示词：

---

### 提示词模板

```
# 角色
你是一位在黎曼几何、信息几何和最优传输理论方面的专家，同时也精通PyTorch实现。

# 任务
我需要改进一个知识蒸馏项目中双曲空间Sinkhorn距离的实现。我认为当前实现存在理论基础不扎实的问题。

# 背景
1. 项目目标: 使用Sinkhorn距离进行知识蒸馏
2. 当前方法: 将softmax概率映射到Lorentz双曲面
3. 理论疑虑: 概率分布的自然几何是信息几何（Fisher-Rao），而非双曲几何

# 具体请求

## 请求1: 理论评估
请评估以下两个方法的优劣：

### 方法A: Fisher-Rao几何（信息几何）
- 使用Fisher-Rao距离度量概率分布差异
- 在概率单纯形上操作
- 数学公式: d_FR(p,q) = arccos(∑√p_i√q_i)

### 方法B: 改进的双曲几何
- 将log-odds映射到双曲切空间
- 使用Lorentz指数映射
- 在双曲面上计算距离

请分析:
1. 哪种方法理论更严谨？
2. 哪种方法更适合知识蒸馏？
3. 各自的优缺点是什么？

## 请求2: 实现指南
基于你推荐的方法，请提供:

1. 详细的PyTorch实现代码
2. 数学公式到代码的对应说明
3. 数值稳定性的处理技巧
4. 单元测试方法

## 请求3: 理论验证
请提供:
1. 如何验证流形约束？
2. 如何检查距离度量的公理（非负性、对称性、三角不等式）？
3. 如何设计消融实验？

## 请求4: 论文引用
如果某个方法值得深入实现，请提供相关的论文引用和资源链接。

# 附件
我已经在文档中提供了当前实现的详细代码和分析，请参考后回答。
```

---

### 如果要深入研究特定方法

#### 询问Fisher-Rao方法

```
请帮我实现基于Fisher-Rao距离的Sinkhorn知识蒸馏：

要求:
1. 实现Fisher-Rao距离矩阵的高效计算
2. 在Fisher-Rao几何中推导Sinkhorn算法
3. 添加对称性和三角不等式的验证
4. 与原始双曲空间方法进行对比实验

当前代码:
[附上 loss.py 中 Sinkhorn 类的代码]

请提供:
- 完整的PyTorch实现
- 数学推导
- 数值稳定性处理
- 测试用例
```

#### 询问改进的双曲空间

```
请帮我改进当前的双曲空间Sinkhorn实现：

当前问题:
1. 直接将概率作为切向量，缺少理论依据
2. 没有验证流形约束
3. 数值稳定性问题

改进方向:
1. 使用log-odds作为切向量
2. 添加流形约束检查
3. 改进数值稳定性
4. 在双曲空间中重新推导Sinkhorn算法

请提供:
- 改进后的完整实现
- 每个改进点的理论依据
- 如何验证改进的有效性
- 与原始实现的对比分析
```

---

## 📝 总结与建议

### 短期建议（立即可做）

1. **快速验证Fisher-Rao方法**（方案A）
   - 实现简单，理论扎实
   - 可能比双曲空间更适合概率分布
   - 1-2天即可实现并测试

2. **添加对比实验**
   ```python
   # 在现有代码中添加
   if args.use_fisher_rao:
       loss = fisher_rao_sinkhorn(y_s, y_t)
   elif args.use_hyperbolic:
       loss = hyperbolic_sinkhorn(y_s, y_t)
   else:
       loss = euclidean_sinkhorn(y_s, y_t)
   ```

3. **数值稳定性检查**
   - 添加NaN/Inf检测
   - 记录训练中的异常

### 中期建议（1-2周）

1. **深入研究信息几何**
   - 阅读Amari的论文
   - 理解Fisher-Rao距离的物理意义
   - 与其他概率距离度量对比

2. **实现改进的双曲空间**（方案B或E）
   - 使用log-odds表示
   - 添加流形约束验证
   - 对比原始实现

3. **完整的消融实验**
   - 在多个GLUE任务上测试
   - 分析不同数据集的特性
   - 找出最适合的几何空间

### 长期建议（1-2月）

1. **理论研究**
   - 双曲空间中的最优传输理论
   - 概率分布的流形表示
   - 发表论文或技术报告

2. **工程优化**
   - 性能优化（GPU加速）
   - 混合精度训练
   - 大规模实验验证

3. **开源与分享**
   - 清理代码，添加文档
   - 发布实验结果
   - 贡献到开源社区

---

## 🔗 有用的链接和资源

- **信息几何入门**: https://www.springer.com/gp/book/9784431559371
- **双曲机器学习**: https://arxiv.org/abs/2109.03217
- **最优传输**: https://arxiv.org/abs/1803.00567
- **Sinkhorn算法**: https://arxiv.org/abs/1306.0895

---

## ✅ 实现检查清单

在实现任何改进方案时，请确保：

- [ ] 理论基础清晰，有数学推导
- [ ] 代码实现符合数学公式
- [ ] 添加数值稳定性处理
- [ ] 实现单元测试
- [ ] 验证流形约束（如适用）
- [ ] 与原始实现进行对比
- [ ] 在多个数据集上验证
- [ ] 记录实验结果和观察
- [ ] 添加详细的代码注释
- [ ] 编写使用文档

---

**最后建议**：我强烈推荐从**方案A（Fisher-Rao）**开始，因为：
1. 理论基础最扎实
2. 实现相对简单
3. 计算效率高
4. 天然适合概率分布
5. 可以快速验证是否真的需要双曲空间

如果Fisher-Rao效果已经很好，可能不需要双曲空间的复杂性。如果Fisher-Rao不够好，再考虑更复杂的混合方法或改进的双曲空间。

祝你的改进项目顺利！如果有任何问题，欢迎继续咨询。
