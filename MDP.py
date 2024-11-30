from typing import Callable, Any

import numpy as np
import random
from abc import abstractmethod


class MDP():
    """
    Data structure for a Markov Decision Process. In mathematical terms,
    MDPs are sometimes defined in terms of a tuple consisting of the various
    components of the MDP, written (S, A, T, R, gamma):

    gamma: discount factor
    S: state space
    A: action space
    T: transition function
    R: reward function
    TR: sample transition and reward. We will us `TR` later to sample the next
        state and reward given the current state and action: s_prime, r = TR(s, a)
    """
    def __init__(self,
                 gamma: float,
                 S: list[Any],
                 A: list[Any],
                 T: Callable[[Any, Any, Any], float] | np.ndarray,
                 R: Callable[[Any, Any], float] | np.ndarray,
                 TR: Callable[[Any, Any], tuple[Any, float]] = None):
        self.gamma = gamma  # discount factor
        self.S = S          # state space
        self.A = A          # action space

        # reward function R(s, a)
        if type(R) == np.ndarray:
            self.R = lambda s, a: R[s, a]
        else:
            self.R = R

        # transition function T(s, a, s')
        # sample next state and reward given current state and action: s', r = TR(s, a)
        if type(T) == np.ndarray:
            self.T = lambda s, a, s_prime: T[s, a, s_prime]
            self.TR = lambda s, a: (np.random.choice(len(self.S), p=T[s, a]), self.R(s, a)) if not np.all(T[s, a] == 0) else (np.random.choice(len(self.S)), self.R(s, a))
        else:
            self.T = T
            self.TR = TR

    def lookahead(self, U: Callable[[Any], float] | np.ndarray, s: Any, a: Any) -> float:
        if callable(U):
            return self.R(s, a) + self.gamma * np.sum([self.T(s, a, s_prime) * U(s_prime) for s_prime in self.S])
        return self.R(s, a) + self.gamma * np.sum([self.T(s, a, s_prime) * U[i] for i, s_prime in enumerate(self.S)])

    def iterative_policy_evaluation(self, policy: Callable[[Any], Any], k_max: int) -> np.ndarray:
        U = np.zeros(len(self.S))
        for _ in range(k_max):
            U = np.array([self.lookahead(U, s, policy(s)) for s in self.S])
        return U

    def policy_evaluation(self, policy: Callable[[Any], Any]) -> np.ndarray:
        R_prime = np.array([self.R(s, policy(s)) for s in self.S])
        T_prime = np.array([[self.T(s, policy(s), s_prime) for s_prime in self.S] for s in self.S])
        I = np.eye(len(self.S))
        return np.linalg.solve(I - self.gamma * T_prime, R_prime)

    def greedy(self, U: Callable[[Any], float] | np.ndarray, s: Any) -> tuple[float, Any]:
        expected_rewards = [self.lookahead(U, s, a) for a in self.A]
        idx = np.argmax(expected_rewards)
        return self.A[idx], expected_rewards[idx]

    def backup(self, U: Callable[[Any], float] | np.ndarray, s: Any) -> float:
        return np.max([self.lookahead(U, s, a) for a in self.A])

    def randstep(self, s: Any, a: Any) -> tuple[Any, float]:
        return self.TR(s, a)

    def simulate(self, s: Any, policy: Callable[[Any], Any], d: int) -> list[tuple[Any, Any, float]]:  # TODO - Create test
        trajectory = []
        for _ in range(d):
            a = policy(s)
            s_prime, r = self.TR(s, a)
            trajectory.append((s, a, r))
            s = s_prime
        return trajectory
    
    def random_policy(self):
        return lambda s, A=self.A: random.choices(A)[0]