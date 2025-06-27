# stochasticity.py
import numpy as np
from scipy.stats import norm, chi2

# ---------------- 全局常量 ----------------
ETA_V = 0.05          # 电压机会约束违约率
ETA_G = 0.05          # 发电机机会约束违约率
XI    = 0.005         # 方差估计置信度 (0.5 %)
LD    = 0.2           # 相对标准差系数：σ = LD · 负荷

# SND quantiles
Z_V  = norm.ppf(1 - ETA_V)          # ≈ 1.645
Z_G  = norm.ppf(1 - ETA_G)          # 同上
Z_G2 = norm.ppf(1 - ETA_G / 2)      # ≈ 1.96

# 历史样本缓存（避免重复生成）
ERROR_HIST = None     # 类型: np.ndarray | None

# -------------------------------------------------
def chisq_interval(s_var, n, alpha):
    """
    卡方置信区间   (n·s²) / χ²_{1-α},  (n·s²) / χ²_{α}
    Parameters
    ----------
    s_var : float
        样本方差 
    n : int
        样本条数
    alpha : float
        右尾违约率 (α = ξ)
    Returns
    -------
    (lower, upper) : tuple[float, float]
    """
    lower = (n * s_var) / chi2.ppf(1 - alpha, df=n)
    upper = (n * s_var) / chi2.ppf(alpha,     df=n)
    return lower, upper


def create_data(var_true, N):
    """
    根据真方差向量生成随机观测矩阵 (|B| × N)
    """
    var_true = np.asarray(var_true, dtype=float)
    std_true = np.sqrt(var_true)
    obs = np.random.randn(var_true.size, N) * std_true[:, None] # randn第一个元素是节点数目，第二个元素是抽样次数
    # 这种写法也可以实现从正态分布中抽N个样本的功能
    return obs


def init_stochasticity750(BUSES,
                       DATA_DIR="basecase",
                       create_new_data=True,
                       sample_N=750,
                       variance_opt="implicit"):
    """
    生成三组方差向量:
        1. var_true_vector    - 真实方差
        2. var_sample_upper   - 样本方差的卡方上置信界 (worst-case)
        3. var_sample_vector  - 样本方差 (MLE)
    """
    global ERROR_HIST

    bus_set = list(BUSES.keys())

    # ---------- 生成真方差 ----------
    if variance_opt == "implicit":
        loads = np.array([BUSES[b].d_P for b in bus_set], dtype=float)
        var_true_vector = (LD * loads) ** 2  # 误差随机变量的真实方差
    # 注意有平方项，是方差，平方内是标准差。后续create_data函数对其开方构造正态分布。
    # ld是相对标准差系数，即标准差相对于负荷的大小
    elif variance_opt == "explicit":
        f_dict = {"basecase": 0.01, "simplecase": 0.1, "33buscase_pu": 4}
        f = f_dict.get(DATA_DIR, 1.0) # DATA_DIR是输入参数（默认输入了“basecase”），输入与字典对应不上则f取1
        var_true_vector = np.full(len(bus_set), f, dtype=float)
    elif variance_opt == "manual":
        raise ValueError("In manual mode please pass your own variance vector.")
    else:
        raise ValueError("Unknown variance_opt")
    
    # ---------- 生成或复用历史误差 ----------
    if create_new_data or ERROR_HIST is None:
        N = 100
        ERROR_HIST = create_data(var_true_vector, N)           # |B| × N
        # 根据所有点的var_true_vector生成所有点的N个样本（预测误差）
    else:
        N = ERROR_HIST.shape[1] # 复用已有数据
    error_hist = ERROR_HIST

    # ---------- 历史样本方差 ----------
    var_sample_vector = (error_hist ** 2).mean(axis=1) # 第i个节点的var_sample_vector，MLE样本方差


    # ---------- 卡方置信上界 ----------
    intervals = np.array([chisq_interval(s, N, XI) for s in var_sample_vector])
    var_sample_upper = intervals[:, 1]                         # 取 upper bound

    var_diff=var_sample_upper-var_true_vector
    dirac=var_diff*1

    # ---------- 加上dirac生成OOS误差 ----------
    if create_new_data or ERROR_HIST is None:
        N = sample_N
        ERROR_HIST = create_data(var_true_vector+dirac, N)           # |B| × N
        # 根据所有点的var_true_vector生成所有点的N个样本（预测误差）
    else:
        N = ERROR_HIST.shape[1] # 复用已有数据
    error_hist = ERROR_HIST

    return var_true_vector, var_sample_upper, var_sample_vector
# 分别是ERROR_VARIANCES_TRUE(真值，用于模拟检验，比如standard_cc)，ERROR_VARIANCES_WC(worst-case，用于 DR-CC)，ERROR_VARIANCES_SAMPLE(样本估计，用于普通 CC)

