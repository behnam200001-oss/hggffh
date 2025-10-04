import math
import random
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .ai_store import ModelStore
from .drl_learner import DRLearner

class AILearner:
    """
    Multi-armed bandit learner with UCB1, Thompson Sampling, and DRL.
    """

    def __init__(self, arms: List[str], algo: str = "ucb", decay: float = 0.995):
        self.arms = arms
        self.algo = algo.lower()
        self.decay = decay
        self.store = ModelStore()
        self.params = self._defaults()
        self.state = self.store.get_state()
        self._init_arms()

        if self.algo == "drl":
            state_size = 10
            action_size = len(arms)
            self.drl = DRLearner(state_size, action_size)
            self.last_state = None
            self.last_action = None
        else:
            self.drl = None

    def _defaults(self):
        p = self.store.get_params()
        return {
            "ucb_c": float(p.get("ucb_c", 1.2)),
            "reward_hit": float(p.get("reward_hit", 1.0)),
            "reward_near": float(p.get("reward_near", 0.05)),
            "reward_empty": float(p.get("reward_empty", 0.0)),
            "mix_random_ratio": float(p.get("mix_random_ratio", 0.5)),
        }

    def _init_arms(self):
        st = self.state
        if "arms" not in st:
            st["arms"] = {}
        for a in self.arms:
            if a not in st["arms"]:
                st["arms"][a] = {"n": 0, "s": 0.0, "alpha": 1.0, "beta": 1.0, "last": time.time()}
        self.store.set_state(st)
        self.store.save()

    def select_arm(self, t: int) -> str:
        self._decay_all()
        if self.algo == "thompson":
            return self._thompson()
        elif self.algo == "drl":
            return self._drl_select()
        elif self.algo == "ucb":
            return self._ucb(t)
        else:
            # اضافه کردن استراتژی ترکیبی
            if random.random() < 0.1:  # 10% اکتشاف
                return random.choice(self.arms)
            return self._ucb(t)

    def _ucb(self, t: int) -> str:
        st = self.store.get_state()["arms"]
        best = None
        best_ucb = -1e9
        for a in self.arms:
            n = st[a]["n"]
            s = st[a]["s"]
            if n == 0:
                return a
            avg = s / n
            bonus = self.params["ucb_c"] * math.sqrt(math.log(max(t, 2)) / n)
            ucb = avg + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best = a
        return best

    def _thompson(self) -> str:
        st = self.store.get_state()["arms"]
        best = None
        best_sample = -1
        for a in self.arms:
            alpha = max(1e-3, st[a]["alpha"])
            beta = max(1e-3, st[a]["beta"])
            sample = random.betavariate(alpha, beta)
            if sample > best_sample:
                best_sample = sample
                best = a
        return best

    def _drl_select(self) -> str:
        state = self._get_state()
        action = self.drl.act(state)
        self.last_state = state
        self.last_action = action
        return self.arms[action]

    def _get_state(self):
        st = self.store.get_state()["arms"]
        state_vec = []
        for a in self.arms:
            n = st[a]["n"]
            s = st[a]["s"]
            avg_reward = s / max(1, n)
            state_vec.append(n / 1000000)
            state_vec.append(avg_reward)
        state_size = self.drl.state_size if self.drl else 10
        while len(state_vec) < state_size:
            state_vec.append(0.0)
        state_vec = state_vec[:state_size]
        return np.reshape(state_vec, [1, state_size])

    def update(self, arm: str, hits: int, total: int, near: int = 0):
        st = self.store.get_state()
        arm_st = st["arms"][arm]
        
        # بهبود محاسبه reward
        hit_reward = self.params["reward_hit"] * hits
        near_reward = self.params["reward_near"] * near
        empty_penalty = self.params["reward_empty"] * max(0, total - hits - near)
        
        # نرمال‌سازی بر اساس تعداد کلیدهای بررسی شده
        reward = (hit_reward + near_reward + empty_penalty) / max(1, total)
        
        # اضافه کردن پاداش اکتشافی برای استراتژی‌های جدید
        if arm_st["n"] < 1000:
            reward += 0.1 * (1 - arm_st["n"] / 1000)
        
        arm_st["n"] += total
        arm_st["s"] += reward * total
        p = reward
        arm_st["alpha"] += p * total
        arm_st["beta"] += (1 - p) * total
        arm_st["last"] = time.time()
        st["arms"][arm] = arm_st
        self.store.set_state(st)
        self.store.save()

        if self.algo == "drl" and self.last_state is not None:
            next_state = self._get_state()
            done = False
            action_index = self.arms.index(arm)
            self.drl.remember(self.last_state, action_index, reward, next_state, done)
            self.drl.replay()
            self.last_state = None

    def _decay_all(self):
        st = self.store.get_state()
        now = time.time()
        for a in self.arms:
            arm = st["arms"][a]
            dt = max(0.0, now - arm.get("last", now))
            if dt > 5.0:
                arm["n"] *= self.decay
                arm["s"] *= self.decay
                arm["alpha"] = 1.0 + (arm["alpha"] - 1.0) * self.decay
                arm["beta"] = 1.0 + (arm["beta"] - 1.0) * self.decay
                arm["last"] = now
        self.store.set_state(st)
        self.store.save()