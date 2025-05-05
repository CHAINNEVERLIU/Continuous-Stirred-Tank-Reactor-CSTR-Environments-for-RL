# -*- coding: utf-8 -*-
"""
创建两个串联的CSTR仿真环境
"""
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
from copy import deepcopy
import matplotlib.pyplot as plt
import random


class TwoSeriesCSTREnv(gym.Env):
    """
    两个串联的CSTR（连续搅拌反应釜）仿真环境：
        - Reactor 1: 第一个CSTR
        - Reactor 2：第二个CSTR，它以第一个CSTR的出料作为入料
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
    --------------------------------------------------------------------------
    控制目标：
        - 控制C2到设定值
        - 注意：不同的初始状态最终能达到的物理极限稳态是不一样的，所以设定控制目标时请先测试极端情况下的稳态
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    # step 1. 定义固定的动力学参数
    Q = 50  # 反应物进料流量[L/min]
    V1, V2 = 100, 100  # 反应器体积[L]
    Cf = 0.5  # 进料浓度[mol/L]
    Tf = 320  # 进料温度[K]
    Tcf = 370  # 冷却水温度[K]
    k0 = 7.2e10  # 反应速率常数[L/(mol·min)]
    E = 8.314e4  # 活化能[J/mol]
    R = 8.314  # 理想气体常数[J/(mol·K)]
    delta_H = -6.78e4  # 反应热[J/mol]
    rou = 1000  # 反应物进料密度[g/L]
    rou_c = 1000  # 冷却水密度[g/L]
    c_p = 0.239  # 反应物比热[J/(g·K)]
    c_pc = 0.239  # 冷却水比热[J/(g·K)]
    U = 6.6e5  # 热交换系数[J/(m2·min·K)]
    A1, A2 = 8.958, 8.958  # 热交换面积[m2]

    dt = 0.1  # 欧拉法的时间步长[min]
    # --------- 原始物理范围(输入环境的状态和动作均为归一化后的) -----------------
    # 状态空间原始范围
    raw_state_low = np.array([0.0, 273.15, 0.0, 273.15], dtype=np.float32)  # [C1, T1, C2, T2]
    raw_state_high = np.array([0.7, 400.0, 0.7, 400.0], dtype=np.float32)  # [C1, T1, C2, T2]

    # 原始动作空间范围
    raw_action_low = np.array([30.0, 30.0], dtype=np.float32)  # [F1, F2]
    raw_action_high = np.array([250.0, 250.0], dtype=np.float32)  # [F1, F2]

    def __init__(self,
                 render_mode: Optional[str] = None,
                 default_target: float = 0.20,  # 默认目标浓度
                 min_concentration: float = 0.05,
                 max_concentration: float = 0.45,  # 上限不超过进料浓度
                 init_mode: str = "random"):
        super(TwoSeriesCSTREnv, self).__init__()

        self.render_mode = render_mode

        # 定义状态空间 (C1, T1, C2, T2) - 归一化到 [-1, 1]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 定义动作空间 (F1, F2) - 归一化到 [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 当前状态
        self.state = None
        self.initial_state_info = {}

        self.init_mode = init_mode
        if self.init_mode == "random":
            self.init_state = None
        elif self.init_mode == "static":
            # [C1, T1, C2, T2]
            self.init_state = np.array([0.45, 310.0, 0.25, 290.0])

        # 每个episode的最大步数
        self.max_steps = 400
        self.current_step = 0

        # 目标浓度[mol/L]
        self.target_C2 = default_target

        self.min_concentration = min_concentration
        self.max_concentration = max_concentration

        # 记忆属性
        self.last_concentration = None
        self.last_action = None
        self.stable_counter = 0
        self.last_error = None

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
        """将归一化动作反归一化到原始范围"""
        raw_state = self.raw_state_low + (normalized_state + 1.0) * (
                self.raw_state_high - self.raw_state_low) / 2.0
        return raw_state.astype(np.float32)

    def _normalize_action(self, raw_action: np.ndarray) -> np.ndarray:
        """将原始状态归一化到 [-1, 1]"""
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
        生成更智能的初始状态

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

        # 添加额外的约束条件
        # 确保温度梯度合理
        if initial_state[1] < initial_state[3]:
            initial_state[1], initial_state[3] = initial_state[3], initial_state[1]
        # 确保浓度梯度合理
        if initial_state[0] < initial_state[2]:
            initial_state[0], initial_state[2] = initial_state[2], initial_state[0]

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
        self.stable_counter = 0
        self.last_error = None

        # 生成初始状态
        if self.init_mode == "random":
            initial_state = self.generate_initial_state()
        elif self.init_mode == "static":
            # [C1, T1, C2, T2]
            initial_state = self.init_state
            noise = np.random.uniform(
                [-0.05, -10, -0.05, -10],
                [0.05, 10, 0.05, 10],
                size=initial_state.shape
            )
            initial_state += noise
        else:
            raise ValueError(f"init_mode={self.init_mode} is not supported, please choose 'random' or 'static'")

        # 记录初始状态信息
        self.initial_state_info = {
            'initial_concentration_1': initial_state[0],
            'initial_temperature_1': initial_state[1],
            'initial_concentration_2': initial_state[2],
            'initial_temperature_2': initial_state[3]
        }
        self.current_step = 0

        # 更新当前状态
        self.state = self._normalize_state(initial_state)

        return self.state.astype(np.float32), self.initial_state_info  # 返回归一化状态和info

    def compute_reward(self, state, action):
        """
        更复杂的奖励函数设计

        Args:
            state (np.ndarray): 归一化后的当前状态 [C1, T1, C2, T2]
            action (np.ndarray): 归一化后的动作 [F1, F2]

        Returns:
            float: 综合奖励
        """
        # 反归一化状态
        raw_state = self._denormalize_state(state)
        C1, T1, C2, T2 = raw_state
        F1, F2 = self._denormalize_action(action)

        # 1. 浓度控制奖励（主要目标）- 使用指数衰减限制惩罚尺度
        concentration_error = np.abs(C2 - self.target_C2)
        # 使用指数函数，限制惩罚尺度在合理范围内
        normalized_error = concentration_error / (self.max_concentration - self.min_concentration)
        concentration_reward = -5.0 * (normalized_error ** 2) - 2.0 * normalized_error
        # concentration_reward = -5.0 * (1.0 - np.exp(-3.0 * normalized_error))

        # 2. 浓度接近目标的额外奖励（使用连续的奖励函数）
        threshold = 0.05
        if concentration_error < threshold:
            # 当误差小于阈值时，提供平滑的正奖励
            concentration_proximity_reward = (1.0 - concentration_error / threshold)
        else:
            concentration_proximity_reward = 0.0

        # 3. 浓度变化趋势奖励
        if self.last_concentration is not None and self.last_error is not None:
            current_error = C2 - self.target_C2
            prev_error = self.last_concentration - self.target_C2

            # 如果误差在减小（方向正确）
            if np.abs(current_error) < np.abs(prev_error):
                concentration_trend_reward = 0.5
            # 如果误差在增加（方向错误）
            elif np.abs(current_error) > np.abs(prev_error):
                concentration_trend_reward = -0.2
            else:
                concentration_trend_reward = 0.0
        else:
            concentration_trend_reward = 0.0

        self.last_concentration = C2
        self.last_error = C2 - self.target_C2

        # 4. 稳定性奖励 - 长时间保持在目标附近
        stability_threshold = 0.02
        if concentration_error < stability_threshold:
            self.stable_counter += 1
            stability_reward = min(2.0, 0.05 * self.stable_counter)  # 最大2分
        else:
            self.stable_counter = max(0, self.stable_counter - 1)  # 逐渐减少而不是立即归零
            stability_reward = 0.0

        # 5. 温度约束（次要目标）- 使用软约束
        ideal_temp_range = (280, 450)
        temp_penalty = 0.0
        for T, name in zip([T1, T2], ["T1", "T2"]):
            if T < ideal_temp_range[0]:
                # 低温惩罚，随着偏离越远惩罚越大
                deviation = (ideal_temp_range[0] - T) / ideal_temp_range[0]
                temp_penalty -= 0.2 * deviation
            elif T > ideal_temp_range[1]:
                # 高温惩罚，高温更危险，惩罚更大
                deviation = (T - ideal_temp_range[1]) / ideal_temp_range[1]
                temp_penalty -= 0.5 * deviation

        # 6. 动作平滑奖励 - 使用二次惩罚
        if self.last_action is not None:
            action_difference = action - self.last_action
            # 使用二次惩罚，大的变化惩罚更重
            action_smoothness_penalty = max(-1.0, -0.05 * np.sum(action_difference ** 2))
        else:
            action_smoothness_penalty = 0.0
        self.last_action = action.copy()

        # 7. 极端情况惩罚 - 使用软边界
        extreme_penalty = 0.0
        if C2 < 0.005:  # 浓度极低（接近零）
            extreme_penalty -= 1.0 * (1.0 - C2 / 0.005)
        elif C2 > 0.95 * self.max_concentration:  # 浓度接近上限
            extreme_penalty -= 1.0 * ((C2 - 0.95 * self.max_concentration) / (0.05 * self.max_concentration))

        reward = (
                1.0 * concentration_reward +  # 浓度控制（主要目标）
                0.5 * temp_penalty  # 温度约束
        )

        # 额外信息记录
        info = {
            'concentration_reward': concentration_reward,
            'concentration_proximity_reward': concentration_proximity_reward,
            'concentration_trend_reward': concentration_trend_reward,
            'stability_reward': stability_reward,
            'temp_penalty': temp_penalty,
            'action_smoothness_penalty': action_smoothness_penalty,
            'extreme_penalty': extreme_penalty,
            'concentration_error': concentration_error,
            'stable_steps': self.stable_counter
        }

        return reward, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一个环境步骤，注意：输入的action是标准化之后的动作，范围在[-1, 1]"""
        self.current_step += 1

        # 确保动作在合法范围内并反归一化
        normalized_action = np.clip(action, self.action_space.low, self.action_space.high)
        action = self._denormalize_action(normalized_action)
        if self.state is None:
            raise ValueError(f"Please call env.reset() to reset the env first!")
            # 更新状态
        original_state = self._denormalize_state(self.state)
        # 添加状态安全检查
        original_state = np.clip(
            original_state,
            self.raw_state_low,
            self.raw_state_high
        )

        # 尝试计算新状态，添加异常处理
        try:
            C1_new, T1_new, C2_new, T2_new = self._dynamics(state=original_state, action=action)
        except Exception as e:
            # 如果动力学计算出错，返回惩罚性奖励和当前状态
            print(f"Dynamics calculation error: {e}")
            return self.state, -10.0, False, True, {
                "error": str(e),
                "raw_action": action
            }

        original_state_new = np.array([C1_new, T1_new, C2_new, T2_new])
        original_state_new = np.clip(
            original_state_new,
            self.raw_state_low,
            self.raw_state_high
        )
        self.state = self._normalize_state(original_state_new)

        # 使用新的奖励函数
        reward, reward_info = self.compute_reward(self.state, normalized_action)

        # 检查是否完成
        terminated = False  # CSTR是连续过程，通常不会终止

        # 检查是否超过最大步数
        truncated = self.current_step >= self.max_steps

        # 额外信息
        info = {
            "reward": reward,
            "raw_action": action,
            "truncated": truncated,
            "state": self.state,
            "original_state": original_state_new,
            "target_C2": self.target_C2,
            "step": self.current_step
        }

        # 更新info字典
        info.update(reward_info)

        return self.state, reward, terminated, truncated, info

    def _dynamics(self, state: np.ndarray, action: np.ndarray):
        """
        该反应的动力学方程
        注意：输入的state和action都要是实际范围内的，不能是标准化后的,返回的状态也是实际范围内的
        :return:
        """
        C1, T1, C2, T2 = state
        F1, F2 = action

        # 安全性检查
        if np.any(np.isnan(state)) or np.any(np.isnan(action)):
            raise ValueError("检测到非法输入：状态或动作包含NaN")

        # 避免除零错误
        T1 = max(T1, 273.15)  # 设置最小温度
        T2 = max(T2, 273.15)
        F1 = np.clip(F1, 1e-5, 1e5)  # 下限设为极小值避免除零
        F2 = np.clip(F2, 1e-5, 1e5)

        # 指数项安全计算
        def safe_exp(x):
            return np.exp(np.clip(x, -100, 100))  # 限制指数输入范围

        dC1_dt = (self.Q / self.V1) * (self.Cf - C1) - self.k0 * C1 * safe_exp(-self.E / (self.R * T1))
        dT1_dt = (self.Q / self.V1) * (self.Tf - T1) \
                 + ((-self.delta_H * self.k0 * C1) / (self.rou * self.c_p)) * safe_exp(-self.E / (self.R * T1)) \
                 + ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V1)) * F1 * (
                         1 - safe_exp(-(self.U * self.A1) / (F1 * self.rou_c * self.c_pc))) * (
                         self.Tcf - T1)

        dC2_dt = (self.Q / self.V2) * (C1 - C2) - self.k0 * C2 * safe_exp(-self.E / (self.R * T2))
        dT2_dt = (self.Q / self.V2) * (T1 - T2) \
                 + ((-self.delta_H * self.k0 * C2) / (self.rou * self.c_p)) * safe_exp(-self.E / (self.R * T2)) \
                 + ((self.rou_c * self.c_pc) / (self.rou * self.c_p * self.V2)) * F2 * (
                         1 - safe_exp(-(self.U * self.A2) / (F2 * self.rou_c * self.c_pc))) * (
                         self.Tcf - T2)

        C1 += dC1_dt * self.dt
        T1 += dT1_dt * self.dt
        C2 += dC2_dt * self.dt
        T2 += dT2_dt * self.dt

        # 最终安全性检查
        return np.clip(
            [C1, T1, C2, T2],
            self.raw_state_low,
            self.raw_state_high
        )

    def render(self):
        """渲染环境，可以根据需要实现可视化"""
        if self.render_mode == "human":
            # 简单的文本渲染
            raw_state = self._denormalize_state(self.state)
            C1, T1, C2, T2 = raw_state
            print(f"Step: {self.current_step}")
            print(f"Reactor 1: C1={C1:.4f} mol/L, T1={T1:.2f} K")
            print(f"Reactor 2: C2={C2:.4f} mol/L, T2={T2:.2f} K")
            print(f"Target C2: {self.target_C2:.4f} mol/L")
            print(f"Error: {np.abs(C2 - self.target_C2):.4f} mol/L")
            print("-" * 50)
        elif self.render_mode == "rgb_array":
            # 简单图形渲染（需扩展）
            pass


def evaluate_model(model, env, num_episodes=10):
    """
    评估训练好的模型性能
    """
    # 重置环境统计
    episode_rewards = []
    episode_states = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_state = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, done, info = env.step(action)

            # 记录第二个反应器浓度
            # raw_state = env.envs[0]._denormalize_state(obs)
            raw_state = env.env_method("_denormalize_state", obs)
            episode_state.append(raw_state)
            total_reward += reward[0]
            obs = new_obs
        episode_rewards.append(total_reward)
        episode_states.append(np.vstack(episode_state))

    episode_states = np.stack(episode_states, axis=0)
    episode_states = np.squeeze(episode_states)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    state_names = ["Reactor 1 Concentration", "Reactor 1 Temperature", "Reactor 2 Concentration",
                   "Reactor 2 Temperature"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    count = 0

    for i in range(episode_states.shape[-1]):
        state = episode_states[:, :, i]
        # 计算平均浓度和标准差
        mean_state = np.nanmean(state, axis=0)
        std_state = np.nanstd(state, axis=0)
        axes[positions[count][0], positions[count][1]].plot(mean_state, color='blue',
                                                            label='Average ' + str(state_names[count]))
        axes[positions[count][0], positions[count][1]].fill_between(
            range(len(mean_state)),
            mean_state - std_state,
            mean_state + std_state,
            color='lightblue',
            alpha=0.3,
            label='±1 Std Dev'
        )
        if count == 2:
            axes[positions[count][0], positions[count][1]].axhline(y=0.2, color='red', linestyle='--')
        axes[positions[count][0], positions[count][1]].set_title('Average with Standard Deviation')
        axes[positions[count][0], positions[count][1]].legend()
        count += 1

    plt.show()
    plt.close()

    episode_final_state = episode_states[:, -1, :]
    for i in range(episode_final_state.shape[0]):
        print(f"Episode {i+1} 的最终稳态:[C1, T1, C2, T2]={episode_final_state[i]}")
    # 打印评估结果
    print(f"平均回合奖励: {np.mean(episode_rewards)}")
    print(f"奖励标准差: {np.std(episode_rewards)}")

    return episode_rewards, episode_states
