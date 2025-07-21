# 2018 Data-Driven DR-AC-CCOPF 论文笔记

## 1. 论文介绍

**背景**：随着分布式能源（DERs）在配电网中渗透率的提高，节点净负荷功率的不确定性显著增加，给配电系统运营商（DSO）的安全经济运行带来严峻挑战。传统随机优化方法（如机会约束最优潮流）虽能处理不确定性，但其依赖精确的概率分布假设，而实际中预测误差分布往往难以准确获取。为此，本文提出一种**数据驱动的分布鲁棒优化（DRO）框架**，通过有限的历史观测数据构建分布不确定性集合，在最坏情况下保证系统安全。 

**方法**：论文在径向配电网场景下采用LinDistFlow线性化近似，首先构建含机会约束的AC-CCOPF模型，对机会约束进行锥形转化；随后基于卡方分布构造关于历史误差样本方差的不确定集，将模型提升为考虑矩不确定性的分布鲁棒优化（DRO）形式。在In-Sample与Out-of-Sample测试中，观察不同参数变化对运行成本与违约率的影响。


## 2. 辐射状配电网的AC-CCOPF模型

### 2.1 系统建模（基于LinDistFlow近似）
#### 网络拓扑定义：
- **节点集合**： $\mathcal{N}$ （根节点 $0$ ，非根节点 $\mathcal{N}^+ := \mathcal{N} \setminus \{0\}$ ）
- **边集合**： $\mathcal{E}$ （每条边 $i \in \mathcal{E}$ 对应下游节点 $i$ ）

#### 节点参数：
| 符号 | 物理意义 | 约束条件 |
|------|----------|----------|
| $d_i^P, d_i^Q$ | 有功/无功功率需求 | - |
| $v_i$ | 电压幅值 | $v_i \in [v_i^{min}, v_i^{max}]$ |
| $u_i$ | 电压幅值平方 ($u_i = v_i^2$) | $u_i^{min} \leq u_i \leq u_i^{max}$ |
| $g_i^P, g_i^Q$ | 可控DER的有功/无功出力 | $g_i^P \in [g_i^{P,min}, g_i^{P,max}]$<br>$g_i^Q \in [g_i^{Q,min}, g_i^{Q,max}]$ |

#### 线路参数：
| 符号 | 物理意义 |
|------|----------|
| $R_i, X_i$ | 电阻/电抗 |
| $S_i^{max}$ | 视在功率限值 |
| $f_i^P, f_i^Q$ | 有功/无功功率流 |

#### LinDistFlow方程：

$$(d_i^P - g_i^P) + \sum_{j \in \mathcal{C}_i} f_j^P = f_i^P \quad \forall i \in \mathcal{N}^+$$

$$(d_i^Q - g_i^Q) + \sum_{j \in \mathcal{C}_i} f_j^Q = f_i^Q \quad \forall i \in \mathcal{N}^+$$

$$u_{\mathcal{A}_i} - 2(f_i^P R_i + f_i^Q X_i) = u_i \quad \forall i \in \mathcal{N}^+$$

#### 物理约束：

$$\sqrt{(f_i^P)^2 + (f_i^Q)^2} \leq S_i^{max} \quad \forall i \in \mathcal{E}$$

### 2.2 机会约束最优潮流（CC-OPF）

#### 不确定建模：
- **净负荷注入**： $\boldsymbol{d}_i = \tilde{\boldsymbol{d}}_i + \boldsymbol{\epsilon}_i$
  - $\tilde{\boldsymbol{d}}_i$ ：给定预测值
  - $\boldsymbol{\epsilon}_i$ ：预测误差 $\sim \mathcal{N}(0, \sigma_i^2)$

#### 实时控制策略（仿射调整）：

$$\boldsymbol{g}_i^P = \tilde{g}_i^P + \alpha_i \boldsymbol{\hat{\epsilon}}^P, \quad i \in \mathcal{G}$$

$$\boldsymbol{g}_i^Q = \tilde{g}_i^Q + \alpha_i \boldsymbol{\hat{\epsilon}}^Q, \quad i \in \mathcal{G}$$

其中：
- $\boldsymbol{\hat{\epsilon}}^P = \sum_{i \in \mathcal{N}} \boldsymbol{\epsilon}_i^P$
- $\sum_{i \in \mathcal{G}} \alpha_i = 1$

#### 功率传输分布因子（PTDF）：
定义矩阵 $A$ （ $|\mathcal{E}| \times |\mathcal{N}^+|$ ）：

$$a_{ij} = \begin{cases} 
1 & \text{线路} i \text{在根节点到节点} j \text{的路径上} \\
0 & \text{否则}
\end{cases}$$

#### 不确定功率流：

$$f_i^P = \tilde{f}_i^P + a_{is}(\boldsymbol{\epsilon}^P - \alpha \boldsymbol{\hat{\epsilon}}^P)$$

$$f_i^Q = \tilde{f}_i^Q + a_{is}(\boldsymbol{\epsilon}^Q - \alpha \boldsymbol{\hat{\epsilon}}^Q)$$

#### 不确定电压幅值平方：

$$u_i = \tilde{u}_i - 2a_{si}^T \left[ R \circ A(\boldsymbol{\epsilon}^P - \alpha \boldsymbol{\hat{\epsilon}}^P) + X \circ A(\boldsymbol{\epsilon}^Q - \alpha \boldsymbol{\hat{\epsilon}}^Q) \right]$$

#### 完整优化模型：

$$\min_{(\tilde{g}, \alpha, f, \tilde{u})} \mathbb{E}[f(\boldsymbol{g}^P, \boldsymbol{g}^Q, \boldsymbol{\epsilon}^P, \boldsymbol{\epsilon}^Q)]$$
约束条件：

$$\sum_{i \in \mathcal{G}} \alpha_i = 1$$

$$(\tilde{d}_i^P - \tilde{g}_i^P) + \sum_{j \in \mathcal{C}_i} \tilde{f}_j^P = \tilde{f}_i^P \quad \forall i \in \mathcal{E}$$

$$\tilde{u}_{\mathcal{A}_i} - 2(\tilde{f}_i^P R_i + \tilde{f}_i^Q X_i) = \tilde{u}_i \quad \forall i \in \mathcal{N}^+$$

$$\mathbb{P}(g_i^{P,min} \leq \boldsymbol{g}_i^P \leq g_i^{P,max}) \geq 1 - 2\eta_g \quad \forall i \in \mathcal{G}$$

$$\mathbb{P}(g_i^{Q,min} \leq \boldsymbol{g}_i^Q \leq g_i^{Q,max}) \geq 1 - 2\eta_g \quad \forall i \in \mathcal{G}$$

$$\mathbb{P}(u_i^{min} \leq \boldsymbol{u}_i \leq u_i^{max}) \geq 1 - 2\eta_v \quad \forall i \in \mathcal{N}$$


## 3. 数据驱动的DR-AC-CCOPF

### 3.1 问题重定义（从随机优化到分布鲁棒优化）
传统AC-CCOPF假设预测误差分布完全已知（均值和方差精确），但实际中：
- 真实分布 $P$ 未知
- 仅有 $N$ 个历史观测样本： $\mathcal{H}(\epsilon_i) := \{\hat{\epsilon}_{i,t}\}_{t=1}^N$

#### 分布鲁棒优化框架：

$$\min_{\boldsymbol{x}} \sup_{P \in \mathcal{U}} \mathbb{E}_P [f(\boldsymbol{x}, \boldsymbol{\epsilon})]$$

- $\boldsymbol{x}$ ：决策变量（发电计划 $\tilde{g}_i^P$ 、参与因子 $\alpha_i$ 等）
- $\mathcal{U}$ ：基于数据构建的**分布不确定性集合**
- $\sup$ ：针对最坏情况分布优化

### 3.2 不确定性集合构建（基于 $\chi^2$ 分布）

#### 样本方差计算

$$\hat{\sigma}_i^2 = \frac{1}{N} \sum_{t=1}^N \hat{\epsilon}_{i,t}^2, \quad \forall i \in \mathcal{N}$$

#### 方差置信区间
真实方差 $\sigma_i^2$ 的 $(1-\xi)$ 置信区间：

$$\mathcal{U}_{\sigma_i^2} = \left[ \underbrace{\frac{N\hat{\sigma}_i^2}{\chi_{N,1-\xi/2}^2}}_{\hat{\zeta}_{i,l}}, \underbrace{\frac{N\hat{\sigma}_i^2}{\chi_{N,\xi/2}^2}}_{\hat{\zeta}_{i,h}} \right]$$

- $\chi_{N,\xi}^2$ ：自由度为 $N$ 的 $\chi^2$ 分布的 $\xi$ -分位数
- **关键性质**：区间不对称（ $\hat{\zeta}_{i,h} > |\hat{\sigma}_i^2 - \hat{\zeta}_{i,l}|$ ）

### 3.3 最坏情况期望成本

#### 发电成本函数

$$\mathbb{E}[c_i(\boldsymbol{g}_i^P)] = c_{i2} \left( \alpha_i^2 \text{Var}(\boldsymbol{\hat{\epsilon}}^P) + (\tilde{g}_i^P)^2 \right) + c_{i1}\tilde{g}_i^P + c_{i0}$$

#### 全局成本的最坏情况

$$\sup_{\sigma_i^2 \in \mathcal{U}_{\sigma_i^2}} \mathbb{E}[f_P] = \sum_{i \in \mathcal{G}} \left[ c_{i2} \left( \alpha_i^2 \sum_{j \in \mathcal{N}} \hat{\zeta}_{j,h}^P + (\tilde{g}_i^P)^2 \right) + c_{i1}\tilde{g}_i^P + c_{i0} \right]$$

- **核心洞察**：最坏情况对应方差取上界 $\hat{\zeta}_{j,h}^P$

### 3.4 完整DR-AC-CCOPF模型

#### 目标函数

$$\min_{\tilde{g},\alpha,f,\tilde{u}} \sum_{i \in \mathcal{G}} \left[ c_{i2}^P \left( \alpha_i^2 \sum_{j \in \mathcal{N}} \hat{\zeta}_{j,h}^P + (\tilde{g}_i^P)^2 \right) + c_{i1}\tilde{g}_i^P + c_{i0}^P \right]$$

#### 关键约束

#### (1) 功率平衡方程

$$(\tilde{d}_i^P - \tilde{g}_i^P) + \sum_{j \in \mathcal{C}_i} \tilde{f}_j^P = \tilde{f}_i^P \quad \forall i \in \mathcal{E}, p \in \{P,Q\}$$

#### (2) 电压方程（LinDistFlow）

$$\tilde{u}_{\mathcal{A}_i} - 2(\tilde{f}_i^P R_i + \tilde{f}_i^Q X_i) = \tilde{u}_i \quad \forall i \in \mathcal{N}^+$$

#### (3) 发电出力机会约束

$$\tilde{g}_i^{P,max} \geq \tilde{g}_i^P + z_{\eta_g} \alpha_i \sqrt{\sum_{k=1}^{b} \hat{\zeta}_{k,h}^P} \quad \forall i \in \mathcal{G}, p \in \{P,Q\}$$

$$-\tilde{g}_i^{P,min} \geq -\tilde{g}_i^P + z_{\eta_g} \alpha_i \sqrt{\sum_{k=1}^{b} \hat{\zeta}_{k,h}^P} \quad \forall i \in \mathcal{G}, p \in \{P,Q\}$$

#### (4) 电压机会约束

$$u_i^{max} \geq \tilde{u}_i + z_{\eta_v} \cdot 2 \sqrt{h_i(\alpha)} \quad \forall i \in \mathcal{N}$$

$$-u_i^{min} \geq -\tilde{u}_i + z_{\eta_v} \cdot 2 \sqrt{h_i(\alpha)} \quad \forall i \in \mathcal{N}$$

#### 电压方差函数 $h_i(\alpha)$

$$h_i(\alpha) = \sum_{k=1}^{l} a_{ki} \left[ 
\begin{aligned}
&R_k^2 \sum_{j=1}^{b} a_{kj} \left( \hat{\zeta}_{j,h}^P + \alpha_j^2 \sum_{m=1}^{b} \hat{\zeta}_{m,h}^P \right) \\
+ &X_k^2 \sum_{j=1}^{b} a_{kj} \left( \hat{\zeta}_{j,h}^Q + \alpha_j^2 \sum_{m=1}^{b} \hat{\zeta}_{m,h}^Q \right)
\end{aligned}
\right]$$


## 4. 案例研究

### 4.1 不确定性区间生成
- 使用真实分布生成 $N=100$ 个误差样本
- 通过调整 $\chi^2$ 分位数 $\xi$ 获得不同不确定性区间

### 4.2 样本内评估
- 使用真实分布生成750个随机样本
- 改变 $\eta_v$ 获得：
  - 电压约束违反概率变化曲线
  - 成本变化曲线（展示保守性与经济性权衡）

### 4.3 样本外性能测试
- 构造新分布： $\sigma_{\text{new}}^2 = \sigma^2 + \delta \cdot \Delta$
  - $\delta \in \{0, 0.5, 1\}$ 对应偏移程度
  - 实际取值： $\delta_1=0.26, \delta_2=0.55, \delta_3=0.88$
- 生成750个测试样本
- 关键发现：
  - $\delta$ 增大 → 分布偏离增大 → 电压违反概率增加
  - DR-AC-CCOPF在所有场景均满足理论违反概率限制
```
