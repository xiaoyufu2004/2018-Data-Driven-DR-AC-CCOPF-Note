import pandas as pd
import numpy as np
from gurobipy import *
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Reals, Objective, Constraint, minimize, value
from pyomo.opt import SolverFactory
from LinDistFlow import Solve_LinDist
from CC_distflow_deter import Solve_CCLinDist
from stochasticity import init_stochasticity
from stochasticity750 import init_stochasticity750
from CC_distflow_DRO import Solve_DD_DR_CCLinDist
from CC_distflow_DRO_gurobi import Solve_DD_DR_CCLinDist_grb
from CC_distflow_deter_gurobi import Solve_CCLinDist_grb
from result_tester import test_results

class Generator:
    def __init__(self, index, bus_idx, g_P_max, g_Q_max):
        self.index = index
        self.bus_idx = bus_idx
        self.g_P_max = g_P_max
        self.g_Q_max = g_Q_max
        self.cost = 1.00  # Default value

class Bus:
    def __init__(self, index, d_P, d_Q, v_max, v_min):
        self.index = index
        self.is_root = False
        self.d_P = d_P
        self.d_Q = d_Q
        self.v_max = v_max
        self.v_min = v_min
        self.children = []
        self.ancestor = []
        self.generator = None
        
        den = np.sqrt(d_P**2 + d_Q**2)
        self.cosphi = d_P / den if den > 1e-6 else 1.0

        if np.isnan(self.cosphi):
            self.cosphi = 0
            self.tanphi = 0
        else:
            self.cosphi = self.cosphi
            self.tanphi = np.tan(np.arccos(self.cosphi))

class Line:
    def __init__(self, index, to_node, from_node, r, x, s_max):
        self.index = index
        self.to_node = to_node
        self.from_node = from_node
        self.r = r
        self.x = x
        self.b = x / (r**2 + x**2)
        self.s_max = s_max

## 数据处理
nodes_raw=pd.read_csv("nodes.csv")
lines_raw=pd.read_csv("lines.csv")
generators_raw=pd.read_csv("generators.csv")

buses={}
for _, row in nodes_raw.iterrows():
    index = row['index']
    d_P = row['d_P']
    d_Q = row['d_Q']
    v_max = row['v_max']
    v_min = row['v_min']
    newb = Bus(index, d_P, d_Q, v_max, v_min)
    buses[newb.index] = newb # index是节点编号，从0到14

lines = {}
for _, row in lines_raw.iterrows():
    index = row['index']
    from_node = row['from_node']
    to_node = row['to_node']
    r = row['r']
    x = row['x']
    s_max = row['s_max']
    newl = Line(index, to_node, from_node, r, x, s_max)

    buses[newl.from_node].children.append(newl.to_node) # 该馈线的上游节点的子代的该馈线的下游节点
    buses[newl.to_node].ancestor.append(newl.from_node) # 该馈线的下游节点的父代是该馈线的上游节点

    lines[newl.index] = newl # index是馈线编号，从0到13

generators = {}
for _, row in generators_raw.iterrows():
    index = row['index']
    bus_idx = row['node']
    g_P_max = row['p_max']
    g_Q_max = row['q_max']
    cost = row['cost']
    newg = Generator(index, bus_idx, g_P_max, g_Q_max)
    newg.cost = cost # 默认cost为1，根据实际generator更新

    buses[newg.bus_idx].generator = newg # 属于Generator类

    generators[newg.index] = newg # index是generator编号，g1,g2,g3，generator是字典形式，因此索引才能是字符串
    
# Check topology
r = 0
root_bus = None
for b in buses.keys():
    l = len(buses[b].ancestor)
    if l > 1:
        print(f"Warning: Network not Radial (Bus {buses[b].index})") # 辐射状网络，最多有一个父节点
    elif l == 0: 
        buses[b].is_root = True # 没有父节点，则为根节点
        root_bus = b
        r += 1
if r == 0:
    print("Warning: No root detected") # 根节点有且只有一个
    root_bus = 0
elif r > 1:
    print("Warning: More than one root detected")
 
# Radial PTDF
A = np.zeros((len(lines), len(buses))) # 节点支路关联阵。如果支路i在从root节点到节点j的路径上，则[i,j]标记1.
for b in buses.keys():
    a = b
    a=int(a)
    b=int(b)
    while a != root_bus:
        A[a-1,b] = 1
        a = int(buses[a].ancestor[0])

print("Done preparing Data")



########################################
# 3) 主程序（解析 CSV + 调用 Solve_LinDist）
########################################
if __name__ == "__main__":

    # 1) 读 CSV
    nodes_raw = pd.read_csv("nodes.csv")
    lines_raw = pd.read_csv("lines.csv")
    generators_raw = pd.read_csv("generators.csv")

    # 2) 构造 buses / lines / generators
    buses = {}
    for _, row in nodes_raw.iterrows():
        idx = row['index']
        d_P = row['d_P']
        d_Q = row['d_Q']
        vmax = row['v_max']
        vmin = row['v_min']
        buses[idx] = Bus(idx, d_P, d_Q, vmax, vmin)

    lines = {}
    for _, row in lines_raw.iterrows():
        idx = row['index']
        frm = row['from_node']
        to_ = row['to_node']
        r_ = row['r']
        x_ = row['x']
        smax = row['s_max']
        newline = Line(idx, to_, frm, r_, x_, smax)

        # 建立上下级关系
        buses[newline.from_node].children.append(newline.to_node)
        buses[newline.to_node].ancestor.append(newline.from_node)

        lines[idx] = newline

    generators = {}
    for _, row in generators_raw.iterrows():
        idx = row['index']
        bus_idx = row['node']
        pmax = row['p_max']
        qmax = row['q_max']
        cost = row['cost']
        g = Generator(idx, bus_idx, pmax, qmax)
        g.cost = cost
        buses[bus_idx].generator = g
        generators[idx] = g

    # 3) 检测谁是root
    root_count = 0
    root_bus = None
    for b in buses:
        if len(buses[b].ancestor) == 0:
            buses[b].is_root = True
            root_bus = b
            root_count += 1
    if root_count == 0:
        print("No root bus detected!")
    elif root_count > 1:
        print("More than one root bus found!")

    # 4) 调用函数
    # result, bus_df, line_df = Solve_LinDist(buses, lines, generators)
    
    # ld=0.1
    # error_variances=[(ld*buses[b].d_P)**2 for b in list(buses.keys())]
    ERROR_VARIANCES_TRUE, ERROR_VARIANCES_WC, ERROR_VARIANCES_SAMPLE =init_stochasticity(buses)
    # result, bus_df, line_df = Solve_CCLinDist(buses, lines, generators,ERROR_VARIANCES_TRUE)
    # result, bus_df, line_df = Solve_CCLinDist_grb(buses, lines, generators,ERROR_VARIANCES_TRUE)
    # result, bus_df, line_df = Solve_DD_DR_CCLinDist(buses, lines, generators,ERROR_VARIANCES_WC)
    result, bus_df, line_df = Solve_DD_DR_CCLinDist_grb(buses, lines, generators,ERROR_VARIANCES_WC)
    

    print("Result:", result)
    print(bus_df)
    print(line_df)

    _,_,_=init_stochasticity750(buses)
    test_df = test_results(
        bus_df,
        line_df,
        result,
        samples=750,            # Monte-Carlo 次数
        message="DRO-gurobi",   # 可留空
        print_on=True           # 控制终端输出
    )
    print(test_df)
