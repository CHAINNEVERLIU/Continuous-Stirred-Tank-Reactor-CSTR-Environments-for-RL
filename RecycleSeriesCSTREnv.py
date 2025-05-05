# -*- coding: utf-8 -*-
"""
回流型双CSTR仿真环境：
两个串联的CSTR（连续搅拌反应釜）与回流配置
"""
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from copy import deepcopy
import matplotlib.pyplot as plt
import random
import matplotlib
# 设置中文字体，'SimHei'是黑体，可以换成其他支持中文的字体
plt.rcParams['font.family'] = 'SimHei'

# 设置数学符号字体为支持数学符号的字体，如 STIX 或 DejaVu Sans
plt.rcParams['mathtext.fontset'] = 'stix'  # 或者 'DejaVu Sans'

# 禁用 Unicode 负号以避免显示为方块
plt.rcParams['axes.unicode_minus'] = False


class RecycleTwoSeriesCSTREnv(gym.Env):
    """
    回流型双CSTR（连续搅拌反应釜）仿真环境：
        - Reactor 1: 第一个CSTR，接收新鲜进料和来自Reactor 2的回流
        - Reactor 2：第二个CSTR，它以第一个CSTR的出料作为入料，部分出料回流至Reactor 1
    --------------------------------------------------------------------------
    状态空间：
        - C1：Reactor 1 的出口反应物浓度[mol/L]
        - T1：Reactor 1 的反应器温度 [K]
        - C2：Reactor 2 的出口反应物浓度[mol/L]
        - T2：Reactor 2 的反应器温度 [K]
    --------------------------------------------------------------------------
    动作空间：
        - F1: Reactor 1 的冷却水流量[L/min]
        - F2: Reactor 2 的冷却水流量[L/min]
        - R: 回流比例 [0-0.9]，表示Reactor 2出料回流到Reactor 1的比例
    --------------------------------------------------------------------------
    控制目标：
        - 控制C2到设定值
        - 最小化操作成本（冷却水、回流泵送成本）
        - 保持系统稳定性和安全性
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # 定义固定的动力学参数
    Q = 50  # 新鲜进料流量[L/min]
    V1, V2 = 100, 100  # 反应器体积[L]
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
    rou = 1000  # 反应物进料密度[g/L]
    rou_c = 1000  # 冷却水密度[g/L]
    c_p = 0.239  # 反应物比热[J/(g·K)]
    c_pc = 0.239  # 冷却水比热[J/(g·K)]
    U = 6.6e5  # 热交换系数[J/(m2·min·K)]
    A1, A2 = 8.958, 8.958  # 热交换面积[m2]

    # 欧拉法的时间步长[min]
    dt = 0.1

    # 操作成本参数
    cooling_water_cost = 0.01  # 冷却水成本[$/L]
    recycle_pump_cost = 0.02  # 回流泵送成本[$/L]
    raw_material_cost = 5.0  # 原料成本[$/mol]
    product_value = 10.0  # 产品价值[$/mol]

    # 状态空间原始范围
    raw_state_low = np.array([0.0, 273.15, 0.0, 273.15], dtype=np.float32)  # [C1, T1, C2, T2]
    raw_state_high = np.array([0.7, 450.0, 0.7, 450.0], dtype=np.float32)  # [C1, T1, C2, T2]

    # 原始动作空间范围
    raw_action_low = np.array([30.0, 30.0, 0.0], dtype=np.float32)  # [F1, F2, R]
    raw_action_high = np.array([250.0, 250.0, 0.9], dtype=np.float32)  # [F1, F2, R]

    def __init__(self,
                 render_mode: Optional[str] = None,
                 default_target: float = 0.20,  # 默认目标浓度
                 min_concentration: float = 0.05,
                 max_concentration: float = 0.45,  # 上限不超过进料浓度
                 init_mode: str = "random",
                 enable_side_reaction: bool = True,  # 是否启用副反应
                 enable_catalyst_decay: bool = True,  # 是否启用催化剂失活
                 process_params: Optional[Dict] = None):  # 可自定义的工艺参数
        """
        初始化回流型双CSTR环境

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
        super(RecycleTwoSeriesCSTREnv, self).__init__()

        self.render_mode = render_mode

        # 定义状态空间 (C1, T1, C2, T2) - 归一化到 [-1, 1]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 定义动作空间 (F1, F2, R) - 归一化到 [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
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
            # [C1, T1, C2, T2]
            self.init_state = np.array([0.45, 310.0, 0.25, 290.0])
        elif self.init_mode == "fixed":
            # 将在set_initial_state方法中设置
            self.init_state = None

        # 每个episode的最大步数
        self.max_steps = 400
        self.current_step = 0

        # 目标浓度[mol/L]
        self.target_C2 = default_target

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        # 反应和催化剂参数
        self.enable_side_reaction = enable_side_reaction
        self.enable_catalyst_decay = enable_catalyst_decay
        self.catalyst_activity = 1.0  # 催化剂初始活性

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
        self.prev_T1 = None
        self.prev_T2 = None
        self.accumulated_cost = 0.0
        self.accumulated_production = 0.0
        self.byproduct_amount = 0.0  # 副产物累积量

    def set_initial_state(self, initial_state):
        """
        设置确定的初始状态，用于公平评估

        Args:
            initial_state: 初始状态 [C1, T1, C2, T2]
        """
        self.init_mode = "fixed"
        self.init_state = initial_state

    def get_target(self):
        """
        获取当前目标浓度
        """
        return self.target_C2

    def set_target(self, target):
        """
        设置浓度控制目标

        Args:
            target (float): 目标浓度值

        Returns:
            bool: 是否设置成功
        """
        if self.min_concentration <= target <= self.max_concentration:
            self.target_C2 = target
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
            concentration_range (tuple): 浓度范围 [C1, C2]
            temperature_range (tuple): 温度范围 [T1, T2]
            randomness_factor (float): 随机性程度

        Returns:
            np.ndarray: 初始状态 [C1, T1, C2, T2]
        """
        # 使用self.np_random代替np.random
        if self.np_random is None:
            self.seed()  # 如果未初始化，则使用默认种子

        # 基础初始状态生成
        initial_state = np.array([
            # C1: 第一个反应器浓度
            np.random.uniform(concentration_range[0], concentration_range[1]),

            # T1: 第一个反应器温度
            np.random.uniform(temperature_range[0], temperature_range[1]),

            # C2: 第二个反应器浓度（通常比C1低）
            np.random.uniform(concentration_range[0], concentration_range[1] * 0.8),

            # T2: 第二个反应器温度
            np.random.uniform(temperature_range[0], temperature_range[1])
        ])

        # 添加随机扰动
        noise = np.random.uniform(
            -randomness_factor,
            randomness_factor,
            size=initial_state.shape
        )
        initial_state += noise

        # 添加约束条件
        if initial_state[1] < initial_state[3]:
            initial_state[1], initial_state[3] = initial_state[3], initial_state[1]

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
        self.prev_T1 = None
        self.prev_T2 = None

        # 重置累积变量
        self.accumulated_cost = 0.0
        self.accumulated_production = 0.0
        self.byproduct_amount = 0.0

        # 重置催化剂活性
        self.catalyst_activity = 1.0

        # 生成初始状态
        if self.init_mode == "random":
            initial_state = self.generate_initial_state()
        elif self.init_mode == "static":
            # [C1, T1, C2, T2]
            initial_state = self.init_state.copy()
            noise = np.random.uniform(
                [-0.05, -10, -0.05, -10],
                [0.05, 10, 0.05, 10],
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
            'initial_concentration_1': initial_state[0],
            'initial_temperature_1': initial_state[1],
            'initial_concentration_2': initial_state[2],
            'initial_temperature_2': initial_state[3]
        }
        self.current_step = 0

        # 记录上一步温度，用于计算变化率
        self.prev_T1 = initial_state[1]
        self.prev_T2 = initial_state[3]

        # 更新当前状态
        self.state = self._normalize_state(initial_state)

        return self.state.astype(np.float32), self.initial_state_info

    def compute_reward(self, state, action):
        """
        计算奖励函数，考虑目标控制、操作成本和系统稳定性

        Args:
            state (np.ndarray): 归一化后的当前状态 [C1, T1, C2, T2]
            action (np.ndarray): 归一化后的动作 [F1, F2, R]

        Returns:
            tuple: (奖励值, 信息字典)
        """
        # 反归一化状态和动作
        raw_state = self._denormalize_state(state)
        C1, T1, C2, T2 = raw_state
        raw_action = self._denormalize_action(action)
        F1, F2, recycle_ratio = raw_action

        # 1. 计算当前误差
        current_error = abs(C2 - self.target_C2)
        self.error_history.append(current_error)

        # 2. 记录初始误差(如果是第一步)
        if self.current_step == 1:
            self.initial_error = current_error
            self.last_error = self.initial_error

        # 3. 相对进步奖励(与初始状态相比)
        if self.initial_error > 0:
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
        time_factor = self.current_step / 100.0
        steady_state_penalty = -10.0 * time_factor * (current_error ** 2)

        # 6. 温度约束和安全性相关惩罚
        ideal_temp_range = (280, 380)
        temp_penalty = 0.0
        for T in [T1, T2]:
            if T < ideal_temp_range[0]:
                deviation = (ideal_temp_range[0] - T) / ideal_temp_range[0]
                temp_penalty -= 0.2 * deviation
            elif T > ideal_temp_range[1]:
                deviation = (T - ideal_temp_range[1]) / ideal_temp_range[1]
                temp_penalty -= 0.5 * deviation  # 高温更危险，惩罚更大

        # 7. 温度变化率惩罚
        temp_rate_penalty = 0.0
        if self.prev_T1 is not None and self.prev_T2 is not None:
            T1_rate = abs((T1 - self.prev_T1) / self.dt)
            T2_rate = abs((T2 - self.prev_T2) / self.dt)
            # 温度变化超过5K/min受到惩罚
            if T1_rate > 5.0:
                temp_rate_penalty -= 0.2 * (T1_rate - 5.0)
            if T2_rate > 5.0:
                temp_rate_penalty -= 0.2 * (T2_rate - 5.0)

        # 8. 操作成本惩罚
        # 冷却水成本
        cooling_cost = -0.01 * (F1 + F2)

        # 回流泵送成本
        recycle_cost = -0.05 * recycle_ratio * self.Q

        # 高回流比额外惩罚(高回流比会降低新鲜进料处理能力)
        high_recycle_penalty = 0.0
        if recycle_ratio > 0.7:
            high_recycle_penalty = -0.5 * (recycle_ratio - 0.7) / 0.2

        # 9. 动作平滑奖励
        if self.last_action is not None:
            action_difference = action - self.last_action
            action_smoothness_penalty = -0.1 * np.sum(action_difference ** 2)
        else:
            action_smoothness_penalty = 0.0
        self.last_action = action.copy()

        # 10. 经济效益奖励(产品产出)
        # 计算产品流量
        product_flow = self.Q * (1 - recycle_ratio)
        # 产品产出价值
        product_value_reward = 0.02 * product_flow * C2

        # 11. 更新记忆变量
        self.last_error = current_error
        self.prev_T1 = T1
        self.prev_T2 = T2

        # 更新稳定计数器
        stability_threshold = 0.02
        if current_error < stability_threshold:
            self.stable_counter += 1
        else:
            self.stable_counter = max(0, self.stable_counter - 1)

        # 12. 总奖励计算
        reward = (
                relative_reward +  # 相对初始状态的改进
                step_reward +  # 相对上一步的改进
                steady_state_penalty +  # 稳态误差惩罚
                0.5 * temp_penalty +  # 温度约束
                0.5 * temp_rate_penalty +  # 温度变化率约束
                0.3 * cooling_cost +  # 冷却水成本
                0.3 * recycle_cost +  # 回流泵送成本
                0.3 * high_recycle_penalty +  # 高回流比惩罚
                0.2 * action_smoothness_penalty +  # 动作平滑
                0.3 * product_value_reward  # 产品价值
        )

        # 13. 更新累积变量
        self.accumulated_cost += -(cooling_cost + recycle_cost)
        self.accumulated_production += product_flow * C2 * self.dt

        # 额外信息记录
        info = {
            'relative_reward': relative_reward,
            'step_reward': step_reward,
            'steady_state_penalty': steady_state_penalty,
            'temp_penalty': temp_penalty,
            'temp_rate_penalty': temp_rate_penalty,
            'cooling_cost': cooling_cost,
            'recycle_cost': recycle_cost,
            'high_recycle_penalty': high_recycle_penalty,
            'action_smoothness_penalty': action_smoothness_penalty,
            'product_value_reward': product_value_reward,
            'current_error': current_error,
            'initial_error': self.initial_error,
            'improvement_ratio': 1.0 - (current_error / self.initial_error) if self.initial_error > 0 else 0,
            'stable_steps': self.stable_counter,
            'accumulated_cost': self.accumulated_cost,
            'accumulated_production': self.accumulated_production,
            'byproduct_amount': self.byproduct_amount,
            'catalyst_activity': self.catalyst_activity
        }

        return reward, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个环境步骤"""
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
            F1, F2, recycle_ratio = raw_action
            C1_new, T1_new, C2_new, T2_new = self._dynamics(
                state=original_state,
                cooling_flows=[F1, F2],
                recycle_ratio=recycle_ratio
            )
        except Exception as e:
            # 动力学计算出错，返回惩罚
            print(f"Dynamics calculation error: {e}")
            return self.state, -10.0, False, True, {
                "error": str(e),
                "raw_action": raw_action
            }

        original_state_new = np.array([C1_new, T1_new, C2_new, T2_new])
        original_state_new = np.clip(
            original_state_new,
            self.raw_state_low,
            self.raw_state_high
        )
        self.state = self._normalize_state(original_state_new)

        # 使用奖励函数
        reward, reward_info = self.compute_reward(self.state, normalized_action)

        # 检查是否完成 (CSTR通常不会终止)
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
            "target_C2": self.target_C2,
            "step": self.current_step
        }

        # 更新info字典
        info.update(reward_info)

        return self.state, reward, terminated, truncated, info

    def _dynamics(self, state: np.ndarray, cooling_flows: list, recycle_ratio: float):
        """
        回流型双CSTR动力学方程

        Args:
            state: [C1, T1, C2, T2] - 两个反应器的浓度和温度
            cooling_flows: [F1, F2] - 两个反应器的冷却水流量
            recycle_ratio: 回流比例，CSTR 2出料回流到CSTR 1的比例

        Returns:
            更新后的状态 [C1_new, T1_new, C2_new, T2_new]
        """
        C1, T1, C2, T2 = state
        F1, F2 = cooling_flows

        # 安全检查和参数设置
        T1 = max(T1, 273.15)
        T2 = max(T2, 273.15)
        F1 = np.clip(F1, 1e-5, 1e5)
        F2 = np.clip(F2, 1e-5, 1e5)
        recycle_ratio = np.clip(recycle_ratio, 0.0, 0.9)  # 限制回流比例上限

        # 计算流量
        Q_recycle = self.Q * recycle_ratio  # 回流流量
        Q_product = self.Q * (1 - recycle_ratio)  # 产品流量
        Q_total = self.Q + Q_recycle  # CSTR 1的总进料流量

        # 安全的指数计算
        def safe_exp(x):
            return np.exp(np.clip(x, -100, 100))

        # 应用催化剂活性修正
        k_forward = self.k0_forward * self.catalyst_activity
        k_reverse = self.k0_reverse * self.catalyst_activity
        k_side = self.k0_side * self.catalyst_activity if self.enable_side_reaction else 0.0

        # 计算主反应和副反应速率
        r1_main = k_forward * C1 * safe_exp(-self.E_forward / (self.R * T1)) - \
                  k_reverse * (self.Cf - C1) * safe_exp(-self.E_reverse / (self.R * T1))
        r1_side = k_side * C1 * safe_exp(-self.E_side / (self.R * T1)) if self.enable_side_reaction else 0.0

        r2_main = k_forward * C2 * safe_exp(-self.E_forward / (self.R * T2)) - \
                  k_reverse * (self.Cf - C2) * safe_exp(-self.E_reverse / (self.R * T2))
        r2_side = k_side * C2 * safe_exp(-self.E_side / (self.R * T2)) if self.enable_side_reaction else 0.0

        # CSTR 1的质量平衡 (考虑回流)
        # 鲜料 + 回流 - 出料 - 主反应 - 副反应
        dC1_dt = (self.Q * self.Cf + Q_recycle * C2 - Q_total * C1) / self.V1 - r1_main - r1_side

        # CSTR 1的能量平衡 (考虑回流)
        # 鲜料热量 + 回流热量 - 出料热量 - 反应热 + 冷却热交换
        dT1_dt = (self.Q * self.Tf + Q_recycle * T2 - Q_total * T1) / self.V1 + \
                 ((-self.delta_H_main * r1_main - self.delta_H_side * r1_side) / (self.rou * self.c_p)) + \
                 ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V1)) * F1 * \
                 (1 - safe_exp(-(self.U * self.A1) / (F1 * self.rou_c * self.c_pc))) * (self.Tcf - T1)

        # CSTR 2的质量平衡
        # CSTR 1出料 - 回流 - 产品 - 反应
        dC2_dt = (Q_total * C1 - Q_recycle * C2 - Q_product * C2) / self.V2 - r2_main - r2_side

        # CSTR 2的能量平衡
        # CSTR 1出料热量 - 回流热量 - 产品热量 - 反应热 + 冷却热交换
        dT2_dt = (Q_total * T1 - Q_recycle * T2 - Q_product * T2) / self.V2 + \
                 ((-self.delta_H_main * r2_main - self.delta_H_side * r2_side) / (self.rou * self.c_p)) + \
                 ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V2)) * F2 * \
                 (1 - safe_exp(-(self.U * self.A2) / (F2 * self.rou_c * self.c_pc))) * (self.Tcf - T2)

        # 更新催化剂活性 (缓慢失活)
        if self.enable_catalyst_decay:
            # 催化剂失活与温度、时间相关
            decay_rate = 1e-4 * (1 + 0.05 * max(0, T1 - 320) + 0.05 * max(0, T2 - 320))
            self.catalyst_activity = max(0.5, self.catalyst_activity - decay_rate * self.dt)

        # 记录副产物生成
        self.byproduct_amount += (r1_side * self.V1 + r2_side * self.V2) * self.dt

        # 使用欧拉法更新状态
        C1 += dC1_dt * self.dt
        T1 += dT1_dt * self.dt
        C2 += dC2_dt * self.dt
        T2 += dT2_dt * self.dt

        # 安全性检查
        return np.clip([C1, T1, C2, T2], self.raw_state_low, self.raw_state_high)

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # 文本渲染
            raw_state = self._denormalize_state(self.state)
            C1, T1, C2, T2 = raw_state

            # 获取最后一个动作（如果有）
            recycle_ratio = None
            if self.last_action is not None:
                raw_action = self._denormalize_action(self.last_action)
                recycle_ratio = raw_action[2]

            print(f"\n===== 回流型双CSTR系统状态 (步骤: {self.current_step}) =====")
            print(f"Reactor 1: C1={C1:.4f} mol/L, T1={T1:.2f} K")
            print(f"Reactor 2: C2={C2:.4f} mol/L, T2={T2:.2f} K")
            print(f"目标浓度: {self.target_C2:.4f} mol/L")
            print(f"当前误差: {np.abs(C2 - self.target_C2):.4f} mol/L")

            if recycle_ratio is not None:
                print(f"回流比: {recycle_ratio:.2f}")
                print(f"主流量: {self.Q * (1 - recycle_ratio):.2f} L/min")
                print(f"回流量: {self.Q * recycle_ratio:.2f} L/min")

            print(f"催化剂活性: {self.catalyst_activity:.2f}")
            print(f"累计生产量: {self.accumulated_production:.2f} mol")
            print(f"累计运行成本: {self.accumulated_cost:.2f} $")

            if self.initial_error:
                print(f"相对改进比例: {(1.0 - (abs(C2 - self.target_C2) / self.initial_error)):.2%}")
            print("=" * 60)
        elif self.render_mode == "rgb_array":
            # 图形渲染（需进一步开发）
            pass


def evaluate_model(model, env, num_episodes=10, same_init_state=False):
    """
    改进的模型评估函数，提供多种评估指标

    Args:
        model: 要评估的模型
        env: 环境
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
    steady_state_rewards = []  # 稳态阶段奖励
    improvement_ratios = []  # 相对改进比例
    production_values = []  # 产品产量
    operating_costs = []  # 运行成本
    byproduct_amounts = []  # 副产物量
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
            C2 = raw_state[:, 2]  # 第二个反应器的浓度
            episode_recycle_ratios.append(raw_action[:, 2])  # 记录回流比

            target = None
            if hasattr(env, 'env_method'):
                target = env.env_method("get_target")[0]
            else:
                target = env.target_C2

            errors.append(abs(C2 - target))

            total_reward += reward
            obs = new_obs
            step += 1

        episode_rewards.append(total_reward)
        episode_states.append(np.vstack(episode_state))

        # 记录平均回流比
        recycle_ratios.append(np.mean(episode_recycle_ratios))

        # 记录经济指标
        if hasattr(env, 'env_method'):
            production = env.get_attr("accumulated_production")[0]
            cost = env.get_attr("accumulated_cost")[0]
            byproduct = env.get_attr("byproduct_amount")[0]
        else:
            production = env.accumulated_production
            cost = env.accumulated_cost
            byproduct = env.byproduct_amount

        production_values.append(production)
        operating_costs.append(cost)
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

        # 计算稳态阶段奖励
        if len(converged_steps) > 0:
            steady_idx = converged_steps[0]
            if isinstance(total_reward, (list, np.ndarray)) and len(total_reward) > steady_idx:
                steady_rewards = total_reward[steady_idx:]
                steady_state_rewards.append(np.mean(steady_rewards))
            else:
                # 如果total_reward是标量，无法分割
                steady_state_rewards.append(np.nan)
        else:
            steady_state_rewards.append(np.nan)

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
    print("\n===== 回流型双CSTR系统 - 模型评估结果 =====")
    print(f"平均回合奖励: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"平均收敛步数: {np.mean(convergence_steps):.1f} 步 (阈值: {convergence_threshold})")
    print(f"平均最终误差: {np.mean(final_errors):.4f} mol/L")
    print(f"平均相对改进: {np.mean(improvement_ratios):.2%}")
    print(f"收敛成功率: {(np.array(convergence_steps) < step).mean():.1%}")

    print("\n----- 经济性能指标 -----")
    print(f"平均产品产量: {np.mean(production_values):.2f} mol")
    print(f"平均运行成本: {np.mean(operating_costs):.2f} $")
    print(f"平均利润估计: {np.mean(production_values) * 10.0 - np.mean(operating_costs):.2f} $")
    print(f"平均副产物量: {np.mean(byproduct_amounts):.2f} mol")
    print(f"平均回流比: {np.mean(recycle_ratios):.2%}")

    if not np.all(np.isnan(steady_state_rewards)):
        print(f"平均稳态奖励: {np.nanmean(steady_state_rewards):.4f}")

    # 可视化结果
    visualize_results(episode_states_array, recycle_ratios, production_values, operating_costs)

    return {
        "episode_rewards": episode_rewards,
        "episode_states": episode_states_array,
        "convergence_steps": convergence_steps,
        "final_errors": final_errors,
        "improvement_ratios": improvement_ratios,
        "production_values": production_values,
        "operating_costs": operating_costs,
        "byproduct_amounts": byproduct_amounts,
        "recycle_ratios": recycle_ratios
    }


def visualize_results(episode_states, recycle_ratios=None, production_values=None, operating_costs=None, target_C2=0.2):
    """
    可视化评估结果

    Args:
        episode_states: 形状为 [episodes, steps, features] 的状态数组
        recycle_ratios: 回流比列表
        production_values: 产量列表
        operating_costs: 成本列表
        target_C2: 目标C2浓度
    """
    # 绘制状态变量
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    state_names = ["反应器1浓度(C1)", "反应器1温度(T1)", "反应器2浓度(C2)", "反应器2温度(T2)"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i in range(episode_states.shape[2]):
        # 提取所有回合的第i个状态变量
        state_data = episode_states[:, :, i]

        # 计算平均值和标准差
        mean_state = np.nanmean(state_data, axis=0)
        std_state = np.nanstd(state_data, axis=0)

        row, col = positions[i]
        ax = axes[row, col]

        # 绘制平均值线
        ax.plot(mean_state, color='blue', label='平均值')

        # 绘制标准差区域
        ax.fill_between(
            range(len(mean_state)),
            mean_state - std_state,
            mean_state + std_state,
            color='lightblue',
            alpha=0.3,
            label='±1标准差'
        )

        # 添加标题和标签
        ax.set_title(state_names[i])
        ax.set_xlabel('步数')

        # 对C2添加目标线
        if i == 2:  # C2的索引是2
            ax.axhline(y=target_C2, color='red', linestyle='--', label=f'目标 ({target_C2})')
            ax.set_ylabel('浓度 (mol/L)')
        elif i % 2 == 0:  # C1
            ax.set_ylabel('浓度 (mol/L)')
        else:  # 温度
            ax.set_ylabel('温度 (K)')

        ax.legend()

    plt.tight_layout()
    plt.show()

    # 绘制C2误差收敛图
    plt.figure(figsize=(10, 5))

    # 计算每个时间步的C2误差
    c2_errors = np.abs(episode_states[:, :, 2] - target_C2)
    mean_error = np.nanmean(c2_errors, axis=0)
    std_error = np.nanstd(c2_errors, axis=0)

    plt.plot(mean_error, color='red', label='平均C2误差')
    plt.fill_between(
        range(len(mean_error)),
        np.maximum(0, mean_error - std_error),  # 防止误差变为负值
        mean_error + std_error,
        color='pink',
        alpha=0.3,
        label='±1标准差'
    )

    # 添加收敛阈值线
    plt.axhline(y=0.02, color='green', linestyle='--', label='收敛阈值 (0.02)')

    plt.title('C2浓度控制误差收敛过程')
    plt.xlabel('步数')
    plt.ylabel('C2误差 (mol/L)')
    plt.legend()
    plt.yscale('log')  # 使用对数刻度更容易观察收敛性
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()

    # 绘制经济性能图表（如果提供了数据）
    if production_values is not None and operating_costs is not None:
        plt.figure(figsize=(10, 6))

        # 计算每个回合的利润
        profits = [p * 10.0 - c for p, c in zip(production_values, operating_costs)]

        # 创建柱状图
        bars = plt.bar(range(len(production_values)), profits)

        # 为正利润和负利润使用不同颜色
        for i, profit in enumerate(profits):
            if profit >= 0:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=np.mean(profits), color='blue', linestyle='--', label=f'平均利润: {np.mean(profits):.2f}')

        plt.title('回流型双CSTR系统 - 各回合经济利润')
        plt.xlabel('回合')
        plt.ylabel('利润 ($)')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # 绘制回流比vs利润关系图
        if recycle_ratios is not None:
            plt.figure(figsize=(10, 6))
        plt.scatter(recycle_ratios, profits, c=np.array(production_values),
                    cmap='viridis', s=80, alpha=0.7)

        # 添加趋势线
        if len(recycle_ratios) > 1:
            z = np.polyfit(recycle_ratios, profits, 1)
        p = np.poly1d(z)
        plt.plot(sorted(recycle_ratios), p(sorted(recycle_ratios)),
                 "r--", linewidth=2, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')

        plt.colorbar(label='产品产量 (mol)')
        plt.title('回流比与经济利润关系')
        plt.xlabel('平均回流比')
        plt.ylabel('利润 ($)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()




