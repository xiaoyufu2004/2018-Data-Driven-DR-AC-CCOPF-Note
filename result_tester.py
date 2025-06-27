# mc_tester.py
import sys, math, numpy as np, pandas as pd

# ==============================================================
# 进度条（与 Julia 版视觉一致）
# ==============================================================
def _progress(t: int, n_tot: int, width: int = 50) -> None:
    r = t / n_tot
    n = int(math.floor(r * width))
    bar = "=" * n + " " * (width - n)
    pct = int(r * 100)
    end = "\n" if t == n_tot else ""
    sys.stdout.write(f"\r[{bar}] {pct}%{end}")
    sys.stdout.flush()


# ==============================================================
# 与 Julia exp_arr_quantile 等价 —— 用下标 ⌊n·q⌋
# ==============================================================
def _lower_quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    a_sort = np.sort(arr)
    idx = max(0, int(math.floor(len(a_sort) * q)))
    return a_sort[idx]


# ==============================================================
# Monte-Carlo Tester
# ==============================================================
def test_results(
    bus_df: pd.DataFrame,
    line_df: pd.DataFrame,
    exp_cost: float,
    *,
    samples: int = 750,
    message: str = "",
    print_on: bool = True,
):
    """
    Monte-Carlo 检验 Chance / DRO 结果

    Parameters
    ----------
    bus_df, line_df : DataFrame
        求解器返回的结果
    exp_cost : float
        目标函数的期望成本（Solve_* 返回的 obj）
    samples : int
        Monte-Carlo 抽样次数 (<= ERROR_HIST.shape[1])
    """


    from main import (                # 直接使用 main.py 里的全局
        buses,
        lines,
        generators,
        A as rPTDF,                       # |L| × |B|  ( Radial PTDF )
    )
    from stochasticity750 import ERROR_HIST
    if message:
        print(f"[INFO] Testing Results ({message})")

    n_bus = len(bus_df)
    n_line = len(line_df)

    # ---------- 基准数据 ----------
    dP = bus_df["d_P"].to_numpy(float)
    dQ = bus_df["d_Q"].to_numpy(float)
    gP_base = bus_df["g_P"].to_numpy(float)
    gQ_base = bus_df["g_Q"].to_numpy(float)
    alpha = bus_df["alpha"].to_numpy(float)

    # line 阻抗向量
    line_order = [lines[i] for i in sorted(lines)]
    R = np.array([l.r for l in line_order])
    X = np.array([l.x for l in line_order])

    gen_bus = [generators[g].bus_idx for g in generators]

    # ---------- 统计量 ----------
    v_viol = g_viol = f_viol = 0
    v_viol_sample = 0
    costs = np.zeros(samples)

    # ========== Monte-Carlo ==========
    N_obs = ERROR_HIST.shape[1]
    if samples > N_obs:
        print(f"[WARN] samples({samples}) > observations({N_obs}); 截断至 {N_obs}")
        samples = N_obs

    for s in range(samples):
        _progress(s + 1, samples)

        # --- 负荷扰动 ---
        dev_P = ERROR_HIST[:, s]
        tanphi = np.array([buses[b].tanphi for b in range(0, n_bus)])
        dev_Q = dev_P * tanphi

        dev_P_sum = dev_P.sum()
        dev_Q_sum = dev_Q.sum()

        dP_s = dP + dev_P
        dQ_s = dQ + dev_Q
        gP_s = gP_base + alpha * dev_P_sum
        gQ_s = gQ_base + alpha * dev_Q_sum

        # --- LinDistFlow 重新计算潮流 & 电压 ---
        netP = dP_s - gP_s
        netQ = dQ_s - gQ_s

        fP = rPTDF @ netP
        fQ = rPTDF @ netQ
        v_s = 1.0 - 2.0 * (rPTDF.T @ (R * fP + X * fQ))

        # --- 成本 ---
        costs[s] = sum(buses[b].generator.cost * gP_s[b] ** 2 for b in gen_bus)

        # --- 约束检查 ---
        # 发电机
        for b in gen_bus:
            idx = b - 1
            if (
                gP_s[idx] > buses[b].generator.g_P_max
                or gP_s[idx] < 0
                or gQ_s[idx] > buses[b].generator.g_Q_max
                or gQ_s[idx] < -buses[b].generator.g_Q_max
            ):
                g_viol += 1

        # 电压
        v_flag = np.logical_or(v_s > [buses[b].v_max for b in range(0, n_bus)],
                               v_s < [buses[b].v_min for b in range(0, n_bus)])
        v_viol += v_flag.sum()
        if v_flag.any():
            v_viol_sample += 1

        # 线路
        lim_sq = np.array([l.s_max ** 2 for l in line_order])
        if ((fP**2 + fQ**2) > lim_sq).any():
            f_viol += ((fP**2 + fQ**2) > lim_sq).sum()

    # ---------- 统计汇总 ----------
    median_c = np.median(costs)
    mean_c = costs.mean()
    min_c, max_c = costs.min(), costs.max()
    std_c = costs.std()
    q10 = _lower_quantile(costs, 0.1)
    q90 = _lower_quantile(costs, 0.9)

    delta = costs - exp_cost
    median_d = np.median(delta)
    mean_d = delta.mean()

    if print_on:
        print("\n\n++ Test Results with", samples, "samples ++")
        print(f"{(1 - v_viol / (2 * samples * n_bus)) * 100:.2f}% "
              f"of voltage constraints hold ({v_viol} violations)")
        print(f"{(1 - g_viol / (4 * samples * len(gen_bus))) * 100:.2f}% "
              f"of generation constraints hold ({g_viol} violations)")
        print(f"{(1 - f_viol / (samples * n_line)) * 100:.2f}% "
              f"of flow constraints hold ({f_viol} violations)")
        print()
        print(f"Expected Cost: {exp_cost:.4f}, "
              f"Median Costs: {median_c:.4f} (Min: {min_c:.4f}, Max: {max_c:.4f})")
        print(f"Median deviation from expectation: {median_d:.4f}\n")

    # ---------- DataFrame 输出 ----------
    return pd.DataFrame({
        "v_violation":          [int(v_viol)],
        "f_violation":          [int(f_viol)],
        "g_violation":          [int(g_viol)],
        "median_costs":         [median_c],
        "mean_costs":           [mean_c],
        "min_costs":            [min_c],
        "max_costs":            [max_c],
        "std_dev_cost":         [std_c],
        "quantile10_cost":      [q10],
        "quantile90_cost":      [q90],
        "median_delta":         [median_d],
        "mean_delta":           [mean_d],
        "v_violation_sample":   [int(v_viol_sample)],
    })
