# CC_distflow_DRO_gurobi_stable.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import pyomo.environ as pyo
from pyomo.environ import (
    ConcreteModel, Var, NonNegativeReals, Reals,
    Constraint, Objective, minimize, value
)
from pyomo.opt import SolverFactory


def Solve_DD_DR_CCLinDist_grb(buses, lines, generators, wc_variances):
    """
    Distributionally-Robust Chance-Constrained LinDist-Flow
    —— Gurobi QCP （确定性输出）版本 ——
    """

    print("Starting DRO-CC LinDist model  [solver: Gurobi – deterministic]")

    # ---------- 基本集合 ----------
    bus_set = list(buses.keys())
    line_set = list(lines.keys())
    gen_bus  = [generators[g].bus_idx for g in generators]

    root_bus = next(b for b in bus_set if buses[b].is_root)
    lines_to = {lines[l].to_node: lines[l] for l in line_set}

    # ---------- 常量 ----------
    var_sum_P = sum(wc_variances)                       # Σσ_P²
    var_sum_Q = sum(wc_variances[i] * buses[b].tanphi**2
                    for i, b in enumerate(bus_set))     # Σσ_Q²
    var_P = np.array(wc_variances)
    var_Q = np.array([wc_variances[i] * buses[b].tanphi**2
                      for i, b in enumerate(bus_set)])
    
    ETA_V = 0.05          # 电压机会约束违约率
    ETA_G = 0.05          # 发电机机会约束违约率
    z_g = norm.ppf(1 - ETA_G)
    z_v = norm.ppf(1 - ETA_V)

    # ---------- rPTDF ----------
    rPTDF = np.zeros((len(line_set), len(bus_set)))
    for b in bus_set:
        a = b
        while a != root_bus:
            rPTDF[int(a)-1, int(b)] = 1
            a = buses[a].ancestor[0]

    bus_idx  = {b: i for i, b in enumerate(bus_set)}
    line_idx = {l: i for i, l in enumerate(line_set)}

    # ---------- Pyomo 模型 ----------
    m = ConcreteModel("DRO_CC_LinDist_QCP")
    m.BUS   = pyo.Set(initialize=bus_set)
    m.NGEN  = pyo.Set(initialize=gen_bus)

    # ----- 变量 -----
    m.v      = Var(m.BUS, domain=NonNegativeReals)
    m.fp     = Var(m.BUS, domain=Reals)
    m.fq     = Var(m.BUS, domain=Reals)
    m.gp     = Var(m.BUS, domain=NonNegativeReals)
    m.gq     = Var(m.BUS, domain=Reals)
    m.alpha  = Var(m.BUS, domain=NonNegativeReals)   # 误差分配
    m.sigma  = Var(m.BUS, domain=NonNegativeReals)   # σ_b

    # 初值
    alpha0 = 1.0 / len(gen_bus)
    for b in bus_set:
        m.v[b].value = 1.0
        m.alpha[b].value = alpha0 if b in gen_bus else 0.0
        m.sigma[b].value = 0.0

    # ---------- 目标函数 ----------
    eps = 1e-6                                         # tie-breaker
    def obj_rule(m):
        main_cost = sum((m.gp[b]**2 + m.alpha[b]**2 * var_sum_P) *
                        buses[b].generator.cost
                        for b in m.NGEN)
        tiebreak  = eps * sum((b+1) * m.alpha[b] for b in m.NGEN)
        return main_cost + tiebreak
    m.OBJ = Objective(rule=obj_rule, sense=minimize)

    # ---------- 约束 ----------
    # 1) α 和为 1
    m.sum_alpha = Constraint(expr=sum(m.alpha[b] for b in m.BUS) == 1)

    # 2) 根节点
    m.v_root  = Constraint(expr=m.v[root_bus] == 1.0)
    m.fp_root = Constraint(expr=m.fp[root_bus] == 0)
    m.fq_root = Constraint(expr=m.fq[root_bus] == 0)

    # 3) 非发电母线固定 0
    nongens = list(set(bus_set) - set(gen_bus))
    m.alpha_ng0 = Constraint(nongens, rule=lambda m, b: m.alpha[b] == 0)
    m.gp_ng0    = Constraint(nongens, rule=lambda m, b: m.gp[b] == 0)
    m.gq_ng0    = Constraint(nongens, rule=lambda m, b: m.gq[b] == 0)

    # 4) 功率平衡
    def p_bal(m, b):
        return (buses[b].d_P - m.gp[b] +
                sum(m.fp[k] for k in buses[b].children) == m.fp[b])
    def q_bal(m, b):
        return (buses[b].d_Q - m.gq[b] +
                sum(m.fq[k] for k in buses[b].children) == m.fq[b])
    m.p_bal = Constraint(m.BUS, rule=p_bal)
    m.q_bal = Constraint(m.BUS, rule=q_bal)

    # 5) 电压降 & 线路容量
    def volt_drop(m, b):
        if b == root_bus: return pyo.Constraint.Skip
        anc = buses[b].ancestor[0]
        ln  = lines_to[b]
        return m.v[b] == m.v[anc] - 2*(ln.r*m.fp[b] + ln.x*m.fq[b])
    m.v_drop = Constraint(m.BUS, rule=volt_drop)

    def s_cap(m, b):
        if b == root_bus: return pyo.Constraint.Skip
        ln = lines_to[b]
        return m.fp[b]**2 + m.fq[b]**2 <= ln.s_max**2
    m.s_cap = Constraint(m.BUS, rule=s_cap)

    # 6) 发电机机会约束
    def gp_up(m, b):  return m.gp[b] + z_g*m.alpha[b]*np.sqrt(var_sum_P) <= buses[b].generator.g_P_max
    def gp_low(m, b): return m.gp[b] - z_g*m.alpha[b]*np.sqrt(var_sum_P) >= 0
    def gq_up(m, b):  return m.gq[b] + z_g*m.alpha[b]*np.sqrt(var_sum_Q) <= buses[b].generator.g_Q_max
    def gq_low(m, b): return m.gq[b] - z_g*m.alpha[b]*np.sqrt(var_sum_Q) >= -buses[b].generator.g_Q_max
    m.gp_up  = Constraint(m.NGEN, rule=gp_up)
    m.gp_low = Constraint(m.NGEN, rule=gp_low)
    m.gq_up  = Constraint(m.NGEN, rule=gq_up)
    m.gq_low = Constraint(m.NGEN, rule=gq_low)

    # 7) σ_b 二次锥
    def sigma_qc(m, b):
        if b == root_bus:
            return m.sigma[b] == 0
        term_R = sum(
            rPTDF[line_idx[l], bus_idx[b]] * lines[l].r**2 *
            sum(rPTDF[line_idx[l], bus_idx[j]] *
                (var_P[int(j)] + m.alpha[j]**2 * var_sum_P)
                for j in bus_set)
            for l in line_set
        )
        term_X = sum(
            rPTDF[line_idx[l], bus_idx[b]] * lines[l].x**2 *
            sum(rPTDF[line_idx[l], bus_idx[j]] *
                (var_Q[int(j)] + m.alpha[j]**2 * var_sum_Q)
                for j in bus_set)
            for l in line_set
        )
        return m.sigma[b]**2 >= 4*(term_R + term_X)
    m.sigma_qc = Constraint(m.BUS, rule=sigma_qc)

    # 8) 电压机会约束
    def v_up(m, b):
        if b == root_bus: return pyo.Constraint.Skip
        return m.v[b] + z_v*m.sigma[b] <= buses[b].v_max
    def v_low(m, b):
        if b == root_bus: return pyo.Constraint.Skip
        return m.v[b] - z_v*m.sigma[b] >= buses[b].v_min
    m.v_up  = Constraint(m.BUS, rule=v_up)
    m.v_low = Constraint(m.BUS, rule=v_low)

    # ---------- 求解 ----------
    solver = SolverFactory("gurobi")
    solver.options.update({
        "OutputFlag" : 1,
        "Threads"    : 1,      # 关闭并行 → 可重复
        "Seed"       : 42,     # 固定随机性
        "BarConvTol" : 1e-9,   # 更严格
        "Crossover"  : 1       # 内点 → 基解
    })
    results = solver.solve(m, tee=True)

    if results.solver.termination_condition not in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible
    ):
        print("Gurobi failed:", results.solver.termination_condition)
        return None, None, None

    obj_val = value(m.OBJ)
    print("Optimal objective:", obj_val)

    # ---------- 输出 ----------
    bus_rows, line_rows = [], []
    for b in bus_set:
        bus_rows.append([
            b, buses[b].d_P, buses[b].d_Q,
            value(m.gp[b]), value(m.gq[b]), value(m.v[b]), value(m.alpha[b])
        ])
        if b != root_bus:
            fp = value(m.fp[b]); fq = value(m.fq[b])
            v_up = value(m.v[buses[b].ancestor[0]])
            line_rows.append([
                lines_to[b].index, buses[b].ancestor[0], b,
                fp, fq, 0 if v_up == 0 else (fp**2+fq**2)/v_up**2
            ])

    bus_df = pd.DataFrame(bus_rows,
                          columns=["bus","d_P","d_Q","g_P","g_Q","v_sq","alpha"])
    line_df = pd.DataFrame(line_rows,
                           columns=["line","from","to","f_P","f_Q","a_sq"])
    bus_df.sort_values("bus", inplace=True)
    line_df.sort_values("to", inplace=True)

    return obj_val, bus_df, line_df
