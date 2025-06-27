import pandas as pd
import numpy as np
from gurobipy import *
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Reals, Objective, Constraint, minimize, value
from pyomo.opt import SolverFactory
##############################
# 2) 主函数：Solve_LinDist() #
##############################
def Solve_LinDist(buses, lines, generators):
    print("Starting LinDist Model (via Pyomo + IPOPT)")

    # -- 集合(索引)获取 --
    bus_set = list(buses.keys())       # e.g. [0,1,2, ..., 14]
    line_set = list(lines.keys())      # e.g. [0,1,2, ..., 13]
    gen_set = list(generators.keys())  # e.g. ['g1','g2','g3'] or int
    gen_bus = [generators[g].bus_idx for g in generators]

    # 寻找根节点
    root_bus = None
    for b in bus_set:
        if buses[b].is_root:
            root_bus = b
            break

    # 为了方便在下面写约束：根据 to_node 索引出 Line。标签是下游节点，值是支路
    lines_to = { lines[l].to_node : lines[l] for l in lines }

    # ============ 2.1 创建 Pyomo 模型 ============
    m = ConcreteModel("DistFlowPyomo")

    # ============ 2.2 定义变量 ============
    # 电压的平方：非负
    m.v = Var(bus_set, domain=NonNegativeReals)

    # 有功潮流 / 无功潮流：实数
    m.fp = Var(bus_set, domain=Reals)
    m.fq = Var(bus_set, domain=Reals)

    # 有功 / 无功发电量
    m.gp = Var(bus_set, domain=NonNegativeReals)  # >= 0
    m.gq = Var(bus_set, domain=Reals)             # 可以正负

    # ============ 2.3 定义目标函数 ============
    # 你的原模型：Minimize sum((gp + gq) * cost)
    # 如果要分开也可以：sum(gp*cost + gq*cost)
    def objective_rule(m):
        return sum( (m.gp[b] + m.gq[b]) * buses[b].generator.cost
                    for b in gen_bus
                    if buses[b].generator is not None )
    m.obj = Objective(rule=objective_rule, sense=minimize)

    # ============ 2.4 定义约束 ============

    # 2.4.1 发电机上下限约束
    #（A）带发电机的 Bus
    def gp_upper_rule(m, b):
        return m.gp[b] <= buses[b].generator.g_P_max
    def gp_lower_rule(m, b):
        return m.gp[b] >= 0
    def gq_upper_rule(m, b):
        return m.gq[b] <= buses[b].generator.g_Q_max
    def gq_lower_rule(m, b):
        return m.gq[b] >= -buses[b].generator.g_Q_max

    m.gen_bus = pyo.Set(initialize=gen_bus)
    m.gp_upper = Constraint(m.gen_bus, rule=gp_upper_rule)
    m.gp_lower = Constraint(m.gen_bus, rule=gp_lower_rule)
    m.gq_upper = Constraint(m.gen_bus, rule=gq_upper_rule)
    m.gq_lower = Constraint(m.gen_bus, rule=gq_lower_rule)

    #（B）不带发电机的 Bus：gp=0, gq=0
    def no_gen_rule_gp(m, b):
        return m.gp[b] == 0
    def no_gen_rule_gq(m, b):
        return m.gq[b] == 0

    no_gen_buses = set(bus_set) - set(gen_bus)
    m.no_gen_bus = pyo.Set(initialize=no_gen_buses)
    m.no_gen_con_gp = Constraint(m.no_gen_bus, rule=no_gen_rule_gp)
    m.no_gen_con_gq = Constraint(m.no_gen_bus, rule=no_gen_rule_gq)

    # 2.4.2 功率平衡约束
    def p_balance_rule(m, b):
        # buses[b].d_P - gp[b] + sum(fp[k] for k in children) == fp[b]
        return ( buses[b].d_P - m.gp[b] 
                 + sum(m.fp[ch] for ch in buses[b].children ) 
                 == m.fp[b] )
    m.p_balance = Constraint(bus_set, rule=p_balance_rule)

    def q_balance_rule(m, b):
        # buses[b].d_Q - gq[b] + sum(fq[k] for k in children) == fq[b]
        return ( buses[b].d_Q - m.gq[b] 
                 + sum(m.fq[ch] for ch in buses[b].children ) 
                 == m.fq[b] )
    m.q_balance = Constraint(bus_set, rule=q_balance_rule)

    # 2.4.3 电压上下限
    def vmax_rule(m, b):
        return m.v[b] <= buses[b].v_max
    def vmin_rule(m, b):
        return m.v[b] >= buses[b].v_min

    m.vmax_con = Constraint(bus_set, rule=vmax_rule)
    m.vmin_con = Constraint(bus_set, rule=vmin_rule)

    # 2.4.4 非根节点的电压方程 + 线路容量
    def voltage_drop_rule(m, b):
        if b == root_bus:
            return pyo.Constraint.Skip  # 根节点不需要这个约束
        # v[b] = v[b_ancestor] - 2*(r*fp[b] + x*fq[b])
        anc = buses[b].ancestor[0]
        line_ = lines_to[b]
        return m.v[b] == m.v[anc] - 2*( line_.r*m.fp[b] + line_.x*m.fq[b] ) # b：下游节点，anc：上游节点，line_：下游节点的上游支路
    m.vol_drop_con = Constraint(bus_set, rule=voltage_drop_rule)

    def line_capacity_rule(m, b):
        if b == root_bus:
            return pyo.Constraint.Skip
        line_ = lines_to[b]
        return line_.s_max**2 >= m.fp[b]**2 + m.fq[b]**2
    m.line_cap_con = Constraint(bus_set, rule=line_capacity_rule)

    # 2.4.5 根节点的 v/root_flow 约束
    def root_voltage_rule(m):
        return m.v[root_bus] == 1.0  # v_root
    m.root_v = Constraint(rule=root_voltage_rule)

    def root_flow_p_rule(m):
        return m.fp[root_bus] == 0
    m.root_flow_p = Constraint(rule=root_flow_p_rule)

    def root_flow_q_rule(m):
        return m.fq[root_bus] == 0
    m.root_flow_q = Constraint(rule=root_flow_q_rule)

    # ============ 2.5 调用 IPOPT 求解 ============
    solver = SolverFactory('ipopt')  # 你可以设置 solver.options 来调参
    results = solver.solve(m, tee=True)  # tee=True 可以查看 IPOPT详细日志

    # ============ 2.6 处理/返回结果 ============
    # 判断求解状态
    status = results.solver.status
    termination = results.solver.termination_condition

    if (status == pyo.SolverStatus.ok) and (termination == pyo.TerminationCondition.optimal):
        print("IPOPT found an optimal solution.")

        # 准备输出：bus_results, line_results
        bus_results = []
        line_results = []

        for b in bus_set:
            gp_val = value(m.gp[b])
            gq_val = value(m.gq[b])
            v_val  = value(m.v[b])

            row = [b, buses[b].d_P, buses[b].d_Q, gp_val, gq_val, v_val]
            bus_results.append(row)

            # 线路电流计算
            if b != root_bus:
                fp_val = value(m.fp[b])
                fq_val = value(m.fq[b])
                v_anc  = value(m.v[buses[b].ancestor[0]])
                a_res  = 0 if v_anc == 0 else (fp_val**2 + fq_val**2)/v_anc
                line_row = [
                    lines_to[b].index,  # line index
                    buses[b].ancestor[0],  # from
                    b,                    # to
                    fp_val, fq_val, a_res
                ]
                line_results.append(line_row)

        import pandas as pd
        bus_df = pd.DataFrame(
            bus_results,
            columns=['bus','d_P','d_Q','g_P','g_Q','v_squared']
        )
        line_df = pd.DataFrame(
            line_results,
            columns=['line','from','to','f_P','f_Q','a_squared']
        )

        bus_df.sort_values(by='bus', inplace=True)
        line_df.sort_values(by='to', inplace=True)

        obj_val = value(m.obj)

        return obj_val, bus_df, line_df

    else:
        print(f"Solver Status: {status}")
        print(f"Termination Condition: {termination}")
        print("No optimal solution found or solver failed.")
        return None, None, None
