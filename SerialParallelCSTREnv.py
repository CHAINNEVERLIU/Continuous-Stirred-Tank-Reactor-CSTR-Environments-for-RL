# -*- coding: utf-8 -*-
"""
串并联CSTR(连续搅拌反应釜)系统环境

系统拓扑结构:
    - R1(串联): 接收新鲜进料和R4的回流
    - R2和R3(并联): 接收来自R1的分流出料
    - R4(串联): 接收R2和R3的合并出料，部分出料回流至R1
"""
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import random

# 设置中文字体，'SimHei'是黑体，可以换成其他支持中文的字体
plt.rcParams['font.family'] = 'SimHei'

# 设置数学符号字体为支持数学符号的字体，如 STIX 或 DejaVu Sans
plt.rcParams['mathtext.fontset'] = 'stix'  # 或者 'DejaVu Sans'

# 禁用 Unicode 负号以避免显示为方块
plt.rcParams['axes.unicode_minus'] = False


class SerialParallelCSTREnv(gym.Env):
    """
    串并联CSTR系统环境，结构为:
        - R1(串联): 第一个CSTR，接收新鲜进料和来自R4的回流
        - R2和R3(并联): 两个并联的CSTR，分别接收来自R1的部分出料
        - R4(串联): 最后的CSTR，接收R2和R3的合并出料，并将部分出料回流至R1
    --------------------------------------------------------------------------
    状态空间 [8维]:
        - C1: R1的出口反应物浓度 [mol/L]
        - T1: R1的温度 [K]
        - C2: R2的出口反应物浓度 [mol/L]
        - T2: R2的温度 [K]
        - C3: R3的出口反应物浓度 [mol/L]
        - T3: R3的温度 [K]
        - C4: R4的出口反应物浓度 [mol/L]
        - T4: R4的温度 [K]
    --------------------------------------------------------------------------
    动作空间 [6维]:
        - F1: R1的冷却水流量 [L/min]0-agent0
        - F2: R2的冷却水流量 [L/min]1
        - F3: R3的冷却水流量 [L/min]2
        - F4: R4的冷却水流量 [L/min]3
        - S: R1到R2的分流比例 [0-1]，表示R1出料流向R2的比例(1-S流向R3)4-agent0
        - R: R4的回流比例 [0-0.9]，表示R4出料回流到R1的比例5
    --------------------------------------------------------------------------
    控制目标:
        - 主要目标: 控制C4到设定值
        - 次要目标: 最小化操作成本(冷却水、回流泵送成本等)
        - 保持系统稳定性和安全性
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # 定义固定的动力学参数
    Q = 100  # 新鲜进料流量[L/min]
    V1, V2, V3, V4 = 150, 100, 100, 120  # 各反应器体积[L]
    Cf = 0.5  # 进料浓度[mol/L]
    Tf = 320  # 进料温度[K]
    Tcf = 370  # 冷却水温度[K]

    # 反应动力学参数
    k0_forward = 7.2e10  # 正反应速率常数[L/(mol·min)]
    k0_reverse = 3.6e9  # 逆反应速率常数[L/(mol·min)]
    k0_side = 1.8e9  # 副反应速率常数[L/(mol·min)]
    E_forward = 8.314e4  # 正反应活化能[J/mol]
    E_reverse = 9.0e4  # 逆反应活化能[J/mol]
    E_side = 7.5e4  # 副反应活化能[J/mol]
    R = 8.314  # 理想气体常数[J/(mol·K)]

    # 热力学参数
    delta_H_main = -6.78e4  # 主反应反应热[J/mol]
    delta_H_side = -4.5e4  # 副反应反应热[J/mol]
    rou = 1000  # 反应物密度[g/L]
    rou_c = 1000  # 冷却水密度[g/L]
    c_p = 0.239  # 反应物比热[J/(g·K)]
    c_pc = 0.239  # 冷却水比热[J/(g·K)]
    U = 6.6e5  # 热交换系数[J/(m2·min·K)]
    A1, A2, A3, A4 = 10.0, 8.0, 8.0, 9.0  # 各反应器热交换面积[m2]

    # 欧拉法的时间步长[min]
    dt = 0.1

    # 操作成本参数
    cooling_water_cost = 0.01  # 冷却水成本[$/L]
    recycle_pump_cost = 0.02  # 回流泵送成本[$/L]
    split_pump_cost = 0.01  # 分流泵送成本[$/L]
    raw_material_cost = 5.0  # 原料成本[$/mol]
    product_value = 12.0  # 产品价值[$/mol]

    # 状态空间原始范围
    raw_state_low = np.array([0.0, 273.15, 0.0, 273.15, 0.0, 273.15, 0.0, 273.15],
                             dtype=np.float32)  # [C1, T1, C2, T2, C3, T3, C4, T4]
    raw_state_high = np.array([0.7, 450.0, 0.7, 450.0, 0.7, 450.0, 0.7, 450.0],
                              dtype=np.float32)  # [C1, T1, C2, T2, C3, T3, C4, T4]

    # 原始动作空间范围
    raw_action_low = np.array([30.0, 30.0, 30.0, 30.0, 0.1, 0.0], dtype=np.float32)  # [F1, F2, F3, F4, S, R]
    raw_action_high = np.array([300.0, 250.0, 250.0, 300.0, 0.9, 0.8], dtype=np.float32)  # [F1, F2, F3, F4, S, R]

    def __init__(
            self,
            render_mode: Optional[str] = None,
            default_target: float = 0.25,  # 默认目标浓度
            min_concentration: float = 0.05,
            max_concentration: float = 0.45,  # 上限不超过进料浓度
            init_mode: str = "random",
            enable_side_reaction: bool = True,  # 是否启用副反应
            enable_catalyst_decay: bool = True,  # 是否启用催化剂失活
            process_params: Optional[Dict] = None):  # 可自定义的工艺参数
        """
        初始化串并联CSTR环境

        Args:
            render_mode: 渲染模式
            default_target: 默认目标浓度
            min_concentration: 最小允许浓度
            max_concentration: 最大允许浓度
            init_mode: 初始化模式 ("random", "static", "fixed")
            enable_side_reaction: 是否启用副反应
            enable_catalyst_decay: 是否启用催化剂失活
            process_params: 可选的自定义工艺参数
        """
        super(SerialParallelCSTREnv, self).__init__()

        self.render_mode = render_mode

        # 定义状态空间 (C1, T1, C2, T2, C3, T3, C4, T4) - 归一化到 [-1, 1]
        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 8, dtype=np.float32),
            high=np.array([1.0] * 8, dtype=np.float32),
            dtype=np.float32
        )

        # 定义动作空间 (F1, F2, F3, F4, S, R) - 归一化到 [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0] * 6, dtype=np.float32),
            high=np.array([1.0] * 6, dtype=np.float32),
            dtype=np.float32
        )

        # 当前状态
        self.state = None
        self.initial_state_info = {}

        # 初始化模式设置
        self.init_mode = init_mode
        if self.init_mode == "random":
            self.init_state = None
        elif self.init_mode == "static":
            # [C1, T1, C2, T2, C3, T3, C4, T4]
            self.init_state = np.array([0.45, 320.0, 0.35, 310.0, 0.30, 315.0, 0.25, 300.0])
        elif self.init_mode == "fixed":
            # 将在set_initial_state方法中设置
            self.init_state = None

        # 每个episode的最大步数
        self.max_steps = 400
        self.current_step = 0

        # 目标浓度[mol/L]
        self.target_C4 = default_target

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        # 反应和催化剂参数
        self.enable_side_reaction = enable_side_reaction
        self.enable_catalyst_decay = enable_catalyst_decay
        self.catalyst_activity = {
            "R1": 1.0, "R2": 1.0, "R3": 1.0, "R4": 1.0
        }  # 各反应器催化剂初始活性

        # 如果提供了自定义工艺参数，更新默认值
        if process_params:
            for key, value in process_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # 记忆属性
        self.last_concentration = None
        self.last_action = None
        self.last_error = None
        self.stable_counter = 0

        # 跟踪变量
        self.initial_error = None
        self.error_history = []
        self.prev_temps = {"T1": None, "T2": None, "T3": None, "T4": None}
        self.accumulated_cost = 0.0
        self.accumulated_production = 0.0
        self.accumulated_profit = 0.0
        self.byproduct_amount = 0.0  # 副产物累积量
        self.flow_rates = {
            "feed": 0.0, "r1_to_r2": 0.0, "r1_to_r3": 0.0,
            "r2_to_r4": 0.0, "r3_to_r4": 0.0,
            "recycle": 0.0, "product": 0.0
        }

    def set_initial_state(self, initial_state):
        """
        设置确定的初始状态，用于公平评估

        Args:
            initial_state: 初始状态 [C1, T1, C2, T2, C3, T3, C4, T4]
        """
        self.init_mode = "fixed"
        self.init_state = initial_state

    def get_target(self):
        """
        获取当前目标浓度
        """
        return self.target_C4

    def set_target(self, target):
        """
        设置浓度控制目标

        Args:
            target (float): 目标浓度值

        Returns:
            bool: 是否设置成功
        """
        if self.min_concentration <= target <= self.max_concentration:
            self.target_C4 = target
            return True
        return False

    def _normalize_state(self, raw_state: np.ndarray) -> np.ndarray:
        """将原始状态归一化到 [-1, 1]"""
        normalized_state = 2.0 * (raw_state - self.raw_state_low) / (self.raw_state_high - self.raw_state_low) - 1.0
        return normalized_state.astype(np.float32)

    def _denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """将归一化状态反归一化到原始范围"""
        raw_state = self.raw_state_low + (normalized_state + 1.0) * (
                self.raw_state_high - self.raw_state_low) / 2.0
        return raw_state.astype(np.float32)

    def _normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        """将原始动作归一化到 [-1, 1]"""
        normalized_action = 2.0 * (raw_action - self.raw_action_low) / (
                self.raw_action_high - self.raw_action_low) - 1.0
        return normalized_action.astype(np.float32)

    def _denormalize_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """将归一化动作反归一化到原始范围"""
        raw_action = self.raw_action_low + (normalized_action + 1.0) * (
                self.raw_action_high - self.raw_action_low) / 2.0
        return raw_action.astype(np.float32)

    def seed(self, seed: Optional[int] = None):
        """
        设置随机数种子

        Args:
            seed (int, optional): 随机数种子

        Returns:
            list: 使用的随机数种子列表
        """
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def generate_initial_state(self,
                               concentration_range=(0.05, 0.45),
                               temperature_range=(280, 380),
                               randomness_factor=0.05):
        """
        生成初始状态

        Args:
            concentration_range (tuple): 浓度范围
            temperature_range (tuple): 温度范围
            randomness_factor (float): 随机性程度

        Returns:
            np.ndarray: 初始状态 [C1, T1, C2, T2, C3, T3, C4, T4]
        """
        # 使用self.np_random代替np.random
        if self.np_random is None:
            self.seed()  # 如果未初始化，则使用默认种子

        # 基础初始状态生成 - 使用级联递减的浓度和温度分布，符合实际化工流程
        initial_state = np.zeros(8)

        # R1的初始状态 - 通常有较高的浓度和温度
        initial_state[0] = np.random.uniform(concentration_range[0] * 1.2, concentration_range[1])  # C1
        initial_state[1] = np.random.uniform(temperature_range[0] * 1.05, temperature_range[1])  # T1

        # R2和R3的初始状态 - 由于并联通常浓度和温度略低于R1
        initial_state[2] = np.random.uniform(concentration_range[0] * 1.1, initial_state[0] * 0.9)  # C2
        initial_state[3] = np.random.uniform(temperature_range[0] * 1.03, initial_state[1] * 0.95)  # T2
        initial_state[4] = np.random.uniform(concentration_range[0] * 1.0, initial_state[0] * 0.85)  # C3
        initial_state[5] = np.random.uniform(temperature_range[0] * 1.0, initial_state[1] * 0.93)  # T3

        # R4的初始状态 - 通常浓度最低，温度也较低
        initial_state[6] = np.random.uniform(concentration_range[0],
                                             min(initial_state[2], initial_state[4]) * 0.95)  # C4
        initial_state[7] = np.random.uniform(temperature_range[0], min(initial_state[3], initial_state[5]) * 0.97)  # T4

        # 添加随机扰动
        noise = np.random.uniform(
            -randomness_factor,
            randomness_factor,
            size=initial_state.shape
        )
        initial_state += initial_state * noise  # 按比例添加扰动

        # 最终剪裁到合理范围
        initial_state = np.clip(
            initial_state,
            self.raw_state_low,
            self.raw_state_high
        )

        return initial_state

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """重置环境到初始状态"""
        # 使用seed初始化随机数生成器
        if seed is not None:
            self.seed(seed)

        # 调用父类的reset方法
        super().reset(seed=seed)

        # 重置记忆属性
        self.last_concentration = None
        self.last_action = None
        self.last_error = None
        self.stable_counter = 0
        self.initial_error = None
        self.error_history = []
        self.prev_temps = {"T1": None, "T2": None, "T3": None, "T4": None}

        # 重置累积变量
        self.accumulated_cost = 0.0
        self.accumulated_production = 0.0
        self.accumulated_profit = 0.0
        self.byproduct_amount = 0.0
        self.flow_rates = {
            "feed": self.Q, "r1_to_r2": 0.0, "r1_to_r3": 0.0,
            "r2_to_r4": 0.0, "r3_to_r4": 0.0,
            "recycle": 0.0, "product": 0.0
        }

        # 重置催化剂活性
        self.catalyst_activity = {
            "R1": 1.0, "R2": 1.0, "R3": 1.0, "R4": 1.0
        }

        # 生成初始状态
        if self.init_mode == "random":
            initial_state = self.generate_initial_state()
        elif self.init_mode == "static":
            initial_state = self.init_state.copy()
            # 添加少量随机扰动使每次初始状态略有不同
            noise = np.random.uniform(
                [-0.03, -5, -0.03, -5, -0.03, -5, -0.03, -5],
                [0.03, 5, 0.03, 5, 0.03, 5, 0.03, 5],
                size=initial_state.shape
            )
            initial_state += noise
        elif self.init_mode == "fixed":
            initial_state = self.init_state.copy()
        else:
            raise ValueError(
                f"init_mode={self.init_mode} is not supported, please choose 'random', 'static', or 'fixed'")

        # 记录初始状态信息
        self.initial_state_info = {
            'C1': initial_state[0], 'T1': initial_state[1],
            'C2': initial_state[2], 'T2': initial_state[3],
            'C3': initial_state[4], 'T3': initial_state[5],
            'C4': initial_state[6], 'T4': initial_state[7]
        }
        self.current_step = 0

        # 记录初始温度，用于计算变化率
        self.prev_temps["T1"] = initial_state[1]
        self.prev_temps["T2"] = initial_state[3]
        self.prev_temps["T3"] = initial_state[5]
        self.prev_temps["T4"] = initial_state[7]

        # 更新当前状态
        self.state = self._normalize_state(initial_state)

        return self.state.astype(np.float32), self.initial_state_info

    def compute_reward(self, state, action):
        """
        计算奖励函数，考虑目标控制、操作成本和系统稳定性

        Args:
            state (np.ndarray): 归一化后的当前状态 [C1, T1, C2, T2, C3, T3, C4, T4]
            action (np.ndarray): 归一化后的动作 [F1, F2, F3, F4, S, R]

        Returns:
            tuple: (奖励值, 信息字典)
        """
        # 反归一化状态和动作
        raw_state = self._denormalize_state(state)
        C1, T1, C2, T2, C3, T3, C4, T4 = raw_state
        raw_action = self._denormalize_action(action)
        F1, F2, F3, F4, split_ratio, recycle_ratio = raw_action

        # 1. 计算当前误差 - 主要控制目标是C4
        current_error = abs(C4 - self.target_C4)
        self.error_history.append(current_error)

        # 2. 记录初始误差(如果是第一步)
        if self.current_step == 1:
            self.initial_error = current_error
            self.last_error = self.initial_error

        # 3. 相对进步奖励(与初始状态相比)
        if self.initial_error is not None and self.initial_error > 0:
            relative_improvement = (self.initial_error - current_error) / self.initial_error
            # 映射到合理的奖励范围
            relative_reward = 3.0 * np.clip(relative_improvement, -1.0, 1.0)
        else:
            relative_reward = 0.0

        # 4. 步间改进奖励(与上一步相比)
        if self.last_error is not None and self.last_error > 0:
            step_improvement = (self.last_error - current_error) / max(0.001, self.last_error)
            # 随时间衰减，前期给更大的奖励
            step_reward = 2.0 * np.clip(step_improvement, -0.5, 0.5) * np.exp(-0.005 * self.current_step)
        else:
            step_reward = 0.0

        # 5. 稳态误差惩罚(随时间权重增加)
        time_factor = min(1.0, self.current_step / 100.0)
        steady_state_penalty = -10.0 * time_factor * (current_error ** 2)

        # 6. 并联分流平衡奖励 - 鼓励合理利用并联反应器
        # 避免极端分流比例(例如，几乎所有流量都进入一个反应器)
        split_balance_reward = 0.0
        if split_ratio < 0.2 or split_ratio > 0.8:
            # 惩罚极端分流
            split_balance_reward = -0.5 * (min(abs(split_ratio - 0.5) - 0.3, 0) ** 2)
        else:
            # 根据浓度差异奖励适当的分流调整
            c_diff = abs(C2 - C3)
            if c_diff > 0.1:
                # 如果并联反应器浓度差异大，奖励朝着平衡方向的分流调整
                if (C2 > C3 and split_ratio < 0.5) or (C3 > C2 and split_ratio > 0.5):
                    split_balance_reward = 0.3
                else:
                    split_balance_reward = -0.1
            else:
                # 浓度接近平衡时的小奖励
                split_balance_reward = 0.1

        # 7. 温度约束和安全性相关惩罚
        ideal_temp_range = (280, 380)
        temp_penalty = 0.0
        for T, name in zip([T1, T2, T3, T4], ["T1", "T2", "T3", "T4"]):
            if T < ideal_temp_range[0]:
                deviation = (ideal_temp_range[0] - T) / ideal_temp_range[0]
                temp_penalty -= 0.2 * deviation
            elif T > ideal_temp_range[1]:
                deviation = (T - ideal_temp_range[1]) / ideal_temp_range[1]
                temp_penalty -= 0.5 * deviation  # 高温更危险，惩罚更大

        # 8. 温度变化率惩罚
        temp_rate_penalty = 0.0
        for curr_T, name in zip([T1, T2, T3, T4], ["T1", "T2", "T3", "T4"]):
            if self.prev_temps[name] is not None:
                T_rate = abs((curr_T - self.prev_temps[name]) / self.dt)
                # 温度变化超过5K/min受到惩罚
                if T_rate > 5.0:
                    temp_rate_penalty -= 0.2 * (T_rate - 5.0)
            self.prev_temps[name] = curr_T

        # 9. 操作成本惩罚
        # 冷却水成本
        cooling_cost = -0.01 * (F1 + F2 + F3 + F4)

        # 回流泵送成本 - 回流比例越高，成本越高
        recycle_cost = -0.05 * recycle_ratio * self.Q

        # 分流装置成本 - 考虑到分流需要额外设备和能源
        split_cost = -0.01 * self.Q

        # 高回流比额外惩罚(高回流比会降低新鲜进料处理能力)
        high_recycle_penalty = 0.0
        if recycle_ratio > 0.6:
            high_recycle_penalty = -0.5 * (recycle_ratio - 0.6) / 0.2

        # 10. 动作平滑奖励
        if self.last_action is not None:
            action_difference = action - self.last_action
            action_smoothness_penalty = -0.1 * np.sum(action_difference ** 2)
        else:
            action_smoothness_penalty = 0.0
        self.last_action = action.copy()

        # 11. 经济效益奖励(产品产出)
        # 计算产品流量
        product_flow = self.Q * (1 - recycle_ratio)
        # 产品产出价值
        product_value_reward = 0.02 * product_flow * C4

        # 更新流量记录
        self.flow_rates["feed"] = self.Q
        self.flow_rates["r1_to_r2"] = self.Q * split_ratio
        self.flow_rates["r1_to_r3"] = self.Q * (1 - split_ratio)
        self.flow_rates["r2_to_r4"] = self.flow_rates["r1_to_r2"]
        self.flow_rates["r3_to_r4"] = self.flow_rates["r1_to_r3"]
        self.flow_rates["recycle"] = self.Q * recycle_ratio
        self.flow_rates["product"] = product_flow

        # 12. 更新记忆变量
        self.last_error = current_error

        # 更新稳定计数器
        stability_threshold = 0.02
        if current_error < stability_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = max(0, self.stable_counter - 1)

        # 13. 总奖励计算
        reward = (
                relative_reward +  # 相对初始状态的改进
                step_reward +  # 相对上一步的改进
                steady_state_penalty +  # 稳态误差惩罚
                0.3 * split_balance_reward +  # 分流平衡奖励
                0.5 * temp_penalty +  # 温度约束
                0.5 * temp_rate_penalty +  # 温度变化率约束
                0.3 * cooling_cost +  # 冷却水成本
                0.3 * recycle_cost +  # 回流泵送成本
                0.1 * split_cost +  # 分流成本
                0.3 * high_recycle_penalty +  # 高回流比惩罚
                0.2 * action_smoothness_penalty +  # 动作平滑
                0.3 * product_value_reward  # 产品价值
        )

        # 14. 更新累积变量
        # 计算周期成本
        period_cost = -(cooling_cost + recycle_cost + split_cost)
        # 计算周期产值
        period_production = product_flow * C4 * self.dt
        # 计算周期利润
        period_profit = period_production * self.product_value - period_cost

        self.accumulated_cost += period_cost
        self.accumulated_production += period_production
        self.accumulated_profit += period_profit

        # 额外信息记录
        info = {
            'relative_reward': relative_reward,
            'step_reward': step_reward,
            'steady_state_penalty': steady_state_penalty,
            'split_balance_reward': split_balance_reward,
            'temp_penalty': temp_penalty,
            'temp_rate_penalty': temp_rate_penalty,
            'cooling_cost': cooling_cost,
            'recycle_cost': recycle_cost,
            'split_cost': split_cost,
            'high_recycle_penalty': high_recycle_penalty,
            'action_smoothness_penalty': action_smoothness_penalty,
            'product_value_reward': product_value_reward,
            'current_error': current_error,
            'initial_error': self.initial_error,
            'improvement_ratio': 1.0 - (
                        current_error / self.initial_error) if self.initial_error and self.initial_error > 0 else 0,
            'stable_steps': self.stable_counter,
            'accumulated_cost': self.accumulated_cost,
            'accumulated_production': self.accumulated_production,
            'accumulated_profit': self.accumulated_profit,
            'byproduct_amount': self.byproduct_amount,
            'catalyst_activity': self.catalyst_activity,
            'flow_rates': self.flow_rates
        }

        return reward, info

    def _dynamics(self, state: np.ndarray, cooling_flows: list, split_ratio: float, recycle_ratio: float):
        """
        串并联CSTR系统动力学方程

        Args:
            state: [C1, T1, C2, T2, C3, T3, C4, T4] - 四个反应器的浓度和温度
            cooling_flows: [F1, F2, F3, F4] - 四个反应器的冷却水流量
            split_ratio: 分流比例，CSTR 1出料流向CSTR 2的比例
            recycle_ratio: 回流比例，CSTR 4出料回流到CSTR 1的比例

            Returns:
                更新后的状态 [C1_new, T1_new, C2_new, T2_new, C3_new, T3_new, C4_new, T4_new]
        """
        C1, T1, C2, T2, C3, T3, C4, T4 = state
        F1, F2, F3, F4 = cooling_flows

        # 安全检查和参数设置
        T1 = max(T1, 273.15)
        T2 = max(T2, 273.15)
        T3 = max(T3, 273.15)
        T4 = max(T4, 273.15)
        F1 = np.clip(F1, 1e-5, 1e5)  # 避免除零错误
        F2 = np.clip(F2, 1e-5, 1e5)
        F3 = np.clip(F3, 1e-5, 1e5)
        F4 = np.clip(F4, 1e-5, 1e5)
        split_ratio = np.clip(split_ratio, 0.1, 0.9)  # 限制分流比例
        recycle_ratio = np.clip(recycle_ratio, 0.0, 0.8)  # 限制回流比例上限

        # 计算流量
        Q_recycle = self.Q * recycle_ratio  # 回流流量
        Q_product = self.Q * (1 - recycle_ratio)  # 产品流量
        Q_total_in_R1 = self.Q + Q_recycle  # CSTR 1的总进料流量

        # 计算CSTR 1到CSTR 2和CSTR 3的分流
        Q_R1_to_R2 = Q_total_in_R1 * split_ratio
        Q_R1_to_R3 = Q_total_in_R1 * (1 - split_ratio)

        # CSTR 4的总进料流量 = R2出料 + R3出料
        Q_total_in_R4 = Q_R1_to_R2 + Q_R1_to_R3

        # 更新流量记录 (正常操作时返回此值)
        self.flow_rates = {
            "feed": self.Q,
            "r1_to_r2": Q_R1_to_R2,
            "r1_to_r3": Q_R1_to_R3,
            "r2_to_r4": Q_R1_to_R2,
            "r3_to_r4": Q_R1_to_R3,
            "recycle": Q_recycle,
            "product": Q_product
        }

        # 安全的指数计算
        def safe_exp(x):
            return np.exp(np.clip(x, -100, 100))

        # 应用催化剂活性修正
        def get_rate_constants(reactor_id):
            activity = self.catalyst_activity[reactor_id]
            k_forward = self.k0_forward * activity
            k_reverse = self.k0_reverse * activity
            k_side = self.k0_side * activity if self.enable_side_reaction else 0.0
            return k_forward, k_reverse, k_side

        # 获取各反应器的反应速率常数
        k1_forward, k1_reverse, k1_side = get_rate_constants("R1")
        k2_forward, k2_reverse, k2_side = get_rate_constants("R2")
        k3_forward, k3_reverse, k3_side = get_rate_constants("R3")
        k4_forward, k4_reverse, k4_side = get_rate_constants("R4")

        # 计算各反应器的主反应和副反应速率
        # CSTR 1 反应速率
        r1_main = k1_forward * C1 * safe_exp(-self.E_forward / (self.R * T1)) - \
                  k1_reverse * (self.Cf - C1) * safe_exp(-self.E_reverse / (self.R * T1))
        r1_side = k1_side * C1 * safe_exp(-self.E_side / (self.R * T1)) if self.enable_side_reaction else 0.0

        # CSTR 2 反应速率
        r2_main = k2_forward * C2 * safe_exp(-self.E_forward / (self.R * T2)) - \
                  k2_reverse * (self.Cf - C2) * safe_exp(-self.E_reverse / (self.R * T2))
        r2_side = k2_side * C2 * safe_exp(-self.E_side / (self.R * T2)) if self.enable_side_reaction else 0.0

        # CSTR 3 反应速率
        r3_main = k3_forward * C3 * safe_exp(-self.E_forward / (self.R * T3)) - \
                  k3_reverse * (self.Cf - C3) * safe_exp(-self.E_reverse / (self.R * T3))
        r3_side = k3_side * C3 * safe_exp(-self.E_side / (self.R * T3)) if self.enable_side_reaction else 0.0

        # CSTR 4 反应速率
        r4_main = k4_forward * C4 * safe_exp(-self.E_forward / (self.R * T4)) - \
                  k4_reverse * (self.Cf - C4) * safe_exp(-self.E_reverse / (self.R * T4))
        r4_side = k4_side * C4 * safe_exp(-self.E_side / (self.R * T4)) if self.enable_side_reaction else 0.0

        # CSTR 1 的质量平衡 (考虑回流)
        # 鲜料 + 回流 - 出料 - 主反应 - 副反应
        dC1_dt = (self.Q * self.Cf + Q_recycle * C4 - Q_total_in_R1 * C1) / self.V1 - r1_main - r1_side

        # CSTR 1 的能量平衡 (考虑回流)
        # 鲜料热量 + 回流热量 - 出料热量 - 反应热 + 冷却热交换
        dT1_dt = (self.Q * self.Tf + Q_recycle * T4 - Q_total_in_R1 * T1) / self.V1 + \
                 ((-self.delta_H_main * r1_main - self.delta_H_side * r1_side) / (self.rou * self.c_p)) + \
                 ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V1)) * F1 * \
                 (1 - safe_exp(-(self.U * self.A1) / (F1 * self.rou_c * self.c_pc))) * (self.Tcf - T1)

        # CSTR 2 的质量平衡 (接收CSTR 1分流)
        # CSTR 1分流 - 出料 - 反应
        dC2_dt = (Q_R1_to_R2 * C1 - Q_R1_to_R2 * C2) / self.V2 - r2_main - r2_side

        # CSTR 2 的能量平衡
        # CSTR 1分流热量 - 出料热量 - 反应热 + 冷却热交换
        dT2_dt = (Q_R1_to_R2 * T1 - Q_R1_to_R2 * T2) / self.V2 + \
                 ((-self.delta_H_main * r2_main - self.delta_H_side * r2_side) / (self.rou * self.c_p)) + \
                 ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V2)) * F2 * \
                 (1 - safe_exp(-(self.U * self.A2) / (F2 * self.rou_c * self.c_pc))) * (self.Tcf - T2)

        # CSTR 3 的质量平衡 (接收CSTR 1分流)
        # CSTR 1分流 - 出料 - 反应
        dC3_dt = (Q_R1_to_R3 * C1 - Q_R1_to_R3 * C3) / self.V3 - r3_main - r3_side

        # CSTR 3 的能量平衡
        # CSTR 1分流热量 - 出料热量 - 反应热 + 冷却热交换
        dT3_dt = (Q_R1_to_R3 * T1 - Q_R1_to_R3 * T3) / self.V3 + \
                 ((-self.delta_H_main * r3_main - self.delta_H_side * r3_side) / (self.rou * self.c_p)) + \
                 ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V3)) * F3 * \
                 (1 - safe_exp(-(self.U * self.A3) / (F3 * self.rou_c * self.c_pc))) * (self.Tcf - T3)

        # CSTR 4 的质量平衡 (接收CSTR 2和CSTR 3合并出料)
        # CSTR 2出料 + CSTR 3出料 - 回流 - 产品 - 反应
        dC4_dt = (Q_R1_to_R2 * C2 + Q_R1_to_R3 * C3 - Q_recycle * C4 - Q_product * C4) / self.V4 - r4_main - r4_side

        # CSTR 4 的能量平衡
        # CSTR 2出料热量 + CSTR 3出料热量 - 回流热量 - 产品热量 - 反应热 + 冷却热交换
        dT4_dt = (Q_R1_to_R2 * T2 + Q_R1_to_R3 * T3 - Q_recycle * T4 - Q_product * T4) / self.V4 + \
                 ((-self.delta_H_main * r4_main - self.delta_H_side * r4_side) / (self.rou * self.c_p)) + \
                 ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V4)) * F4 * \
                 (1 - safe_exp(-(self.U * self.A4) / (F4 * self.rou_c * self.c_pc))) * (self.Tcf - T4)

        # 更新催化剂活性 (缓慢失活)
        if self.enable_catalyst_decay:
            # 催化剂失活与温度、时间相关
            decay_rates = {
                "R1": 1e-4 * (1 + 0.05 * max(0, T1 - 320)),
                "R2": 1e-4 * (1 + 0.05 * max(0, T2 - 320)),
                "R3": 1e-4 * (1 + 0.05 * max(0, T3 - 320)),
                "R4": 1e-4 * (1 + 0.05 * max(0, T4 - 320))
            }

            for reactor, rate in decay_rates.items():
                self.catalyst_activity[reactor] = max(0.5, self.catalyst_activity[reactor] - rate * self.dt)

        # 记录副产物生成
        self.byproduct_amount += (r1_side * self.V1 + r2_side * self.V2 + r3_side * self.V3 + r4_side * self.V4) * self.dt

        # 使用欧拉法更新状态
        C1 += dC1_dt * self.dt
        T1 += dT1_dt * self.dt
        C2 += dC2_dt * self.dt
        T2 += dT2_dt * self.dt
        C3 += dC3_dt * self.dt
        T3 += dT3_dt * self.dt
        C4 += dC4_dt * self.dt
        T4 += dT4_dt * self.dt

        # 返回更新后的状态
        return np.array([C1, T1, C2, T2, C3, T3, C4, T4])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一个环境步骤

        Args:
            action: 归一化后的动作 [F1, F2, F3, F4, S, R]

        Returns:
            tuple: (新状态, 奖励, 终止标志, 截断标志, 信息字典)
        """
        self.current_step += 1

        # 确保动作在合法范围内并反归一化
        normalized_action = np.clip(action, self.action_space.low, self.action_space.high)
        raw_action = self._denormalize_action(normalized_action)

        if self.state is None:
            raise ValueError(f"Please call env.reset() to reset the env first!")

        # 更新状态
        original_state = self._denormalize_state(self.state)
        # 状态安全检查
        original_state = np.clip(
            original_state,
            self.raw_state_low,
            self.raw_state_high
        )

        # 尝试计算新状态
        try:
            F1, F2, F3, F4, split_ratio, recycle_ratio = raw_action
            new_state = self._dynamics(
                state=original_state,
                cooling_flows=[F1, F2, F3, F4],
                split_ratio=split_ratio,
                recycle_ratio=recycle_ratio
            )
        except Exception as e:
            # 动力学计算出错，返回惩罚
            print(f"Dynamics calculation error: {e}")
            return self.state, -10.0, False, True, {
                "error": str(e),
                "raw_action": raw_action
            }

        original_state_new = np.clip(
            new_state,
            self.raw_state_low,
            self.raw_state_high
        )
        self.state = self._normalize_state(original_state_new)

        # 使用奖励函数
        reward, reward_info = self.compute_reward(self.state, normalized_action)

        # 检查是否完成 (通常不会终止)
        terminated = False

        # 检查是否超过最大步数
        truncated = self.current_step >= self.max_steps

        # 额外信息
        info = {
            "reward": reward,
            "raw_action": raw_action,
            "truncated": truncated,
            "state": self.state,
            "original_state": original_state_new,
            "target_C4": self.target_C4,
            "step": self.current_step
        }

        # 更新info字典
        info.update(reward_info)

        return self.state, reward, terminated, truncated, info

    def render(self):
        """渲染环境，以文本或图形方式显示当前系统状态"""
        if self.render_mode == "human":
            # 文本渲染
            raw_state = self._denormalize_state(self.state)
            C1, T1, C2, T2, C3, T3, C4, T4 = raw_state

            # 获取最后一个动作（如果有）
            split_ratio = None
            recycle_ratio = None
            if self.last_action is not None:
                raw_action = self._denormalize_action(self.last_action)
                _, _, _, _, split_ratio, recycle_ratio = raw_action

            print(f"\n===== 串并联CSTR系统状态 (步骤: {self.current_step}) =====")
            print(f"R1 (入口反应器): C1={C1:.4f} mol/L, T1={T1:.2f} K")
            print(f"R2 (并联反应器1): C2={C2:.4f} mol/L, T2={T2:.2f} K")
            print(f"R3 (并联反应器2): C3={C3:.4f} mol/L, T3={T3:.2f} K")
            print(f"R4 (出口反应器): C4={C4:.4f} mol/L, T4={T4:.2f} K")
            print(f"目标浓度: {self.target_C4:.4f} mol/L")
            print(f"当前误差: {np.abs(C4 - self.target_C4):.4f} mol/L")

            if split_ratio is not None and recycle_ratio is not None:
                print(f"分流比例 (R1→R2): {split_ratio:.2f}")
                print(f"回流比例 (R4→R1): {recycle_ratio:.2f}")
                print(f"进料流量: {self.flow_rates['feed']:.2f} L/min")
                print(f"R1→R2流量: {self.flow_rates['r1_to_r2']:.2f} L/min")
                print(f"R1→R3流量: {self.flow_rates['r1_to_r3']:.2f} L/min")
                print(f"产品流量: {self.flow_rates['product']:.2f} L/min")
                print(f"回流流量: {self.flow_rates['recycle']:.2f} L/min")

            print(f"催化剂活性: R1={self.catalyst_activity['R1']:.2f}, R2={self.catalyst_activity['R2']:.2f}, "
                  f"R3={self.catalyst_activity['R3']:.2f}, R4={self.catalyst_activity['R4']:.2f}")
            print(f"累计生产量: {self.accumulated_production:.2f} mol")
            print(f"累计运行成本: {self.accumulated_cost:.2f} $")
            print(f"累计利润: {self.accumulated_profit:.2f} $")

            if self.initial_error:
                print(f"相对改进比例: {(1.0 - (abs(C4 - self.target_C4) / self.initial_error)):.2%}")
            print("=" * 65)
        elif self.render_mode == "rgb_array":
            # 图形渲染 - 目前未实现
            pass


def visualize_results(episode_states, split_ratios=None, recycle_ratios=None, profits=None, target_C4=0.25):
    """
    可视化评估结果

    Args:
        episode_states: 形状为 [episodes, steps, features] 的状态数组
        split_ratios: 分流比列表
        recycle_ratios: 回流比列表
        profits: 利润列表
        target_C4: 目标C4浓度
    """
    # 准备绘图
    plt.figure(figsize=(18, 12))

    # 1. 绘制所有反应器的浓度趋势
    plt.subplot(2, 2, 1)
    concentrations = ["C1", "C2", "C3", "C4"]
    colors = ['blue', 'green', 'orange', 'red']

    for i, (name, color) in enumerate(zip(concentrations, colors)):
        idx = i * 2  # 因为状态是 [C1, T1, C2, T2, ...]
        mean_conc = np.mean(episode_states[:, :, idx], axis=0)
        std_conc = np.std(episode_states[:, :, idx], axis=0)

        plt.plot(mean_conc, color=color, label=name)
        plt.fill_between(
            range(len(mean_conc)),
            mean_conc - std_conc,
            mean_conc + std_conc,
            color=color,
            alpha=0.2
        )

    # 添加目标线
    plt.axhline(y=target_C4, color='black', linestyle='--', label=f'目标C4 ({target_C4})')

    plt.title('各反应器浓度变化趋势')
    plt.xlabel('步数')
    plt.ylabel('浓度 (mol/L)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 绘制所有反应器的温度趋势
    plt.subplot(2, 2, 2)
    temperatures = ["T1", "T2", "T3", "T4"]

    for i, (name, color) in enumerate(zip(temperatures, colors)):
        idx = i * 2 + 1  # 因为状态是 [C1, T1, C2, T2, ...]
        mean_temp = np.mean(episode_states[:, :, idx], axis=0)
        std_temp = np.std(episode_states[:, :, idx], axis=0)

        plt.plot(mean_temp, color=color, label=name)
        plt.fill_between(
            range(len(mean_temp)),
            mean_temp - std_temp,
            mean_temp + std_temp,
            color=color,
            alpha=0.2
        )

    plt.title('各反应器温度变化趋势')
    plt.xlabel('步数')
    plt.ylabel('温度 (K)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. 绘制C4误差收敛曲线
    plt.subplot(2, 2, 3)
    c4_errors = np.abs(episode_states[:, :, 6] - target_C4)
    mean_error = np.mean(c4_errors, axis=0)
    std_error = np.std(c4_errors, axis=0)

    plt.semilogy(mean_error, color='red', label='C4误差')
    plt.fill_between(
        range(len(mean_error)),
        np.maximum(0.001, mean_error - std_error),  # 避免负值
        mean_error + std_error,
        color='red',
        alpha=0.2
    )

    # 添加收敛阈值线
    plt.axhline(y=0.02, color='green', linestyle='--', label='收敛阈值 (0.02)')

    plt.title('C4浓度控制误差收敛曲线')
    plt.xlabel('步数')
    plt.ylabel('C4误差 (mol/L)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.7)

    # 4. 如果提供了分流比和回流比，绘制它们的关系散点图
    if split_ratios is not None and recycle_ratios is not None and profits is not None:
        plt.subplot(2, 2, 4)

        # 创建散点图，颜色表示利润
        scatter = plt.scatter(split_ratios, recycle_ratios, c=profits,
                              cmap='viridis', s=100, alpha=0.7)

        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('利润 ($)')

        plt.title('分流比与回流比关系图')
        plt.xlabel('分流比 (R1→R2)')
        plt.ylabel('回流比 (R4→R1)')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 标记平均值
        avg_split = np.mean(split_ratios)
        avg_recycle = np.mean(recycle_ratios)
        plt.scatter([avg_split], [avg_recycle], color='red', s=200, marker='*',
                    label=f'平均值 ({avg_split:.2f}, {avg_recycle:.2f})')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 额外绘制一个利润分析图
    if profits is not None:
        plt.figure(figsize=(10, 6))

        # 排序绘制利润柱状图
        sorted_indices = np.argsort(profits)
        sorted_profits = np.array(profits)[sorted_indices]
        indices = np.arange(len(sorted_profits))

        # 使用渐变色显示利润高低
        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sorted_profits)))

        plt.bar(indices, sorted_profits, color=colors)
        plt.axhline(y=np.mean(profits), color='blue', linestyle='--', label=f'平均利润: {np.mean(profits):.2f}')


def evaluate_model(model, env, num_episodes=5, same_init_state=False):
    """
    评估模型在串并联CSTR系统上的性能，提供多种评估指标

    Args:
        model: 要评估的模型
        env: 环境实例
        num_episodes: 评估的回合数
        same_init_state: 是否使用相同的初始状态序列

    Returns:
        dict: 包含各种评估指标的字典
    """
    # 统计变量
    episode_rewards = []
    episode_states = []
    convergence_steps = []  # 记录收敛步数
    final_errors = []  # 记录最终误差
    improvement_ratios = []  # 相对改进比例
    production_values = []  # 产品产量
    operating_costs = []  # 运行成本
    profits = []  # 利润
    byproduct_amounts = []  # 副产物量
    split_ratios = []  # 平均分流比
    recycle_ratios = []  # 平均回流比

    # 如果使用相同初始状态，预先生成
    init_states = None
    if same_init_state and hasattr(env, 'env_method'):
        init_states = []
        temp_env = env
        while hasattr(temp_env, 'env'):
            temp_env = temp_env.env
        for _ in range(num_episodes):
            init_state = temp_env.generate_initial_state()
            init_states.append(init_state)

    for episode in range(num_episodes):
        # 设置初始状态（如果使用相同序列）
        if init_states is not None:
            env.env_method("set_initial_state", init_states[episode])

        obs = env.reset()
        done = False
        total_reward = 0
        episode_state = []
        errors = []  # 记录每步的误差
        episode_split_ratios = []  # 记录每步的分流比
        episode_recycle_ratios = []  # 记录每步的回流比

        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, done, info = env.step(action)

            # 记录状态和误差
            raw_state = None
            if hasattr(env, 'env_method'):
                raw_state = env.env_method("_denormalize_state", obs)[0]
                raw_action = env.env_method("_denormalize_action", action)[0]
            else:
                # 直接访问环境方法
                raw_state = env._denormalize_state(obs)
                raw_action = env._denormalize_action(action)

            episode_state.append(raw_state)
            C4 = raw_state[:, 6]  # 第四个反应器的浓度

            # 记录动作参数
            if raw_action.ndim > 1:
                split_ratio = raw_action[:, 4]  # 分流比
                recycle_ratio = raw_action[:, 5]  # 回流比
            else:
                split_ratio = raw_action[4]  # 分流比
                recycle_ratio = raw_action[5]  # 回流比

            episode_split_ratios.append(split_ratio)
            episode_recycle_ratios.append(recycle_ratio)

            target = None
            if hasattr(env, 'env_method'):
                target = env.env_method("get_target")[0]
            else:
                target = env.target_C4

            errors.append(abs(C4 - target))

            total_reward += reward
            obs = new_obs
            step += 1

            if isinstance(done, bool):
                if done: break
            else:  # 处理向量化环境
                if done[0]: break

        episode_rewards.append(total_reward)
        episode_states.append(np.vstack(episode_state))

        # 记录平均操作参数
        split_ratios.append(np.mean(episode_split_ratios))
        recycle_ratios.append(np.mean(episode_recycle_ratios))

        # 记录经济指标
        if hasattr(env, 'env_method'):
            production = env.get_attr("accumulated_production")[0]
            cost = env.get_attr("accumulated_cost")[0]
            profit = env.get_attr("accumulated_profit")[0]
            byproduct = env.get_attr("byproduct_amount")[0]
        else:
            production = env.accumulated_production
            cost = env.accumulated_cost
            profit = env.accumulated_profit
            byproduct = env.byproduct_amount

        production_values.append(production)
        operating_costs.append(cost)
        profits.append(profit)
        byproduct_amounts.append(byproduct)

        # 计算收敛步数（误差首次小于阈值的步数）
        convergence_threshold = 0.02
        converged_steps = np.where(np.array(errors) < convergence_threshold)[0]
        if len(converged_steps) > 0:
            convergence_steps.append(converged_steps[0])
        else:
            convergence_steps.append(step)  # 未收敛则使用最大步数

        # 计算最终误差
        final_errors.append(errors[-1])

        # 计算相对改进比例
        initial_error = errors[0]
        if initial_error > 0:
            improvement = 1.0 - (errors[-1] / initial_error)
            improvement_ratios.append(improvement)
        else:
            improvement_ratios.append(0.0)

    # 处理episode_states，确保它是3D数组 [episodes, steps, features]
    episode_states_array = None
    try:
        episode_states_array = np.stack(episode_states)
    except:
        # 如果episode有不同长度，使用零填充
        max_len = max(len(s) for s in episode_states)
        padded_states = []
        for s in episode_states:
            if len(s) < max_len:
                pad = np.zeros((max_len - len(s), s.shape[1]))
                padded_states.append(np.vstack((s, pad)))
            else:
                padded_states.append(s)
        episode_states_array = np.stack(padded_states)

    # 输出评估结果
    print("\n===== 串并联CSTR系统 - 模型评估结果 =====")
    print(f"平均回合奖励: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"平均收敛步数: {np.mean(convergence_steps):.1f} 步 (阈值: {convergence_threshold})")
    print(f"平均最终误差: {np.mean(final_errors):.4f} mol/L")
    print(f"平均相对改进: {np.mean(improvement_ratios):.2%}")
    print(f"收敛成功率: {(np.array(convergence_steps) < step).mean():.1%}")

    print("\n----- 经济性能指标 -----")
    print(f"平均产品产量: {np.mean(production_values):.2f} mol")
    print(f"平均运行成本: {np.mean(operating_costs):.2f} $")
    print(f"平均利润: {np.mean(profits):.2f} $")
    print(f"平均副产物量: {np.mean(byproduct_amounts):.2f} mol")
    print(f"平均分流比: {np.mean(split_ratios):.2f}")
    print(f"平均回流比: {np.mean(recycle_ratios):.2f}")

    # 可视化结果
    visualize_results(episode_states_array, split_ratios, recycle_ratios, profits, 0.2)

    return {
        "episode_rewards": episode_rewards,
        "episode_states": episode_states_array,
        "convergence_steps": convergence_steps,
        "final_errors": final_errors,
        "improvement_ratios": improvement_ratios,
        "production_values": production_values,
        "operating_costs": operating_costs,
        "profits": profits,
        "byproduct_amounts": byproduct_amounts,
        "split_ratios": split_ratios,
        "recycle_ratios": recycle_ratios
    }















