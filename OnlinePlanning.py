"""Chapter 9: Online Planning"""

import numpy as np

from MDP import MDP
from abc import abstractmethod, ABC
from typing import Any, Callable
from Solutions import MDPSolutionMethod


class OnlinePlanningMethod(MDPSolutionMethod):
    @abstractmethod
    def __call__(self, s: Any) -> Any:
        pass


def rollout(P: MDP, s: Any, policy: Callable[[Any], Any], d: int) -> float:
    ret = 0.0
    for t in range(d):
        a = policy(s)
        s, r = P.randstep(s, a)
        ret += (P.gamma**t) * r
    return ret


class RolloutLookahead(OnlinePlanningMethod):
    def __init__(self, P: MDP, policy: Callable[[Any], Any], d: int):
        self.P = P            # problem
        self.policy = policy  # rollout policy
        self.d = d            # depth

    def __call__(self, s: Any) -> Any:
        def U(s): return rollout(self.P, s, self.policy, self.d)
        return (self.P.greedy(U, s))[0]


def forward_search(P: MDP, s: Any, d: int, U: Callable[[Any], float]) -> tuple[Any, float]:
    if d <= 0:
        return (None, U(s))
    best_a, best_u = (None, -np.inf)
    def U_prime(s): return (forward_search(P, s, d - 1, U))[1]
    for a in P.A:
        u = P.lookahead(U_prime, s, a)
        if u > best_u:
            best_a, best_u = (a, u)
    return best_a, best_u


class ForwardSearch(OnlinePlanningMethod):
    def __init__(self, P: MDP, d: int, U: Callable[[Any], float]):
        self.P = P  # problem
        self.d = d  # depth
        self.U = U  # value function at depth d

    def __call__(self, s: Any) -> Any:
        return (forward_search(self.P, s, self.d, self.U))[0]


def branch_and_bound(P: MDP, s: Any, d: int,
                     U_lo: Callable[[Any], float],
                     Q_hi: Callable[[Any, Any], float]) -> tuple[Any, float]:
    if d <= 0:
        return (None, U_lo(s))
    def U_prime(s): return branch_and_bound(P, s, d - 1, U_lo, Q_hi)[1]
    best_a, best_u = (None, -np.inf)
    for a in sorted(P.A, key=(lambda a: Q_hi(s, a)), reverse=True):
        if Q_hi(s, a) < best_u:
            return best_a, best_u # safe to prune
        u = P.lookahead(U_prime, s, a)
        if u > best_u:
            best_a, best_u = (a, u)
    return best_a, best_u


class BranchAndBound(OnlinePlanningMethod):
    def __init__(self, P: MDP, d: int, U_lo: Callable[[Any], float], Q_hi: Callable[[Any, Any], float]):
        self.P = P        # problem
        self.d = d        # depth
        self.U_lo = U_lo  # lower bound on value function at depth d
        self.Q_hi = Q_hi  # upper bound on action value function

    def __call__(self, s: Any) -> Any:
        return (branch_and_bound(self.P, s, self.d, self.U_lo, self.Q_hi))[0]


def sparse_sampling(P: MDP, s: Any, d: int, m: int, U: Callable[[Any], float]):
    if d <= 0:
        return (None, U(s))
    best_a, best_u = (None, -np.inf)
    for a in P.A:
        u = 0.0
        for _ in range(m):
            s_prime, r = P.randstep(s, a)
            a_prime, u_prime = sparse_sampling(P, s_prime, d - 1, m, U)
            u += (r + P.gamma * u_prime) / m
        if u > best_u:
            best_a, best_u = (a, u)
    return best_a, best_u


class SparseSampling(OnlinePlanningMethod):
    def __init__(self, P: MDP, d: int, m: int, U: Callable[[Any], float]):
        self.P = P  # problem
        self.d = d  # depth
        self.m = m  # number of samples
        self.U = U  # value function at depth d

    def __call__(self, s: Any) -> Any:
        return (sparse_sampling(self.P, s, self.d, self.m, self.U))[0]


class MonteCarloTreeSearch(OnlinePlanningMethod):
    def __init__(self,
                 P: MDP,
                 N: dict[tuple[Any, Any], int],
                 Q: dict[tuple[Any, Any], float],
                 d: int,
                 m: int,
                 c: float,
                 U: Callable[[Any], float]):
        self.P = P  # problem
        self.N = N  # visit counts
        self.Q = Q  # action value estimates
        self.d = d  # depth
        self.m = m  # number of simulations
        self.c = c  # exploration constant
        self.U = U  # value function estimate

    def __call__(self, s: Any) -> Any:
        for _ in range(self.m):
            self.simulate(s, d=self.d, steps=0)  # Start with steps=0
        return self.P.A[np.argmax([self.Q[(s, a)] for a in self.P.A])]

    def simulate(self, s: Any, d: int, steps: int):
        if d <= 0:
            return self.U(s)
        if (s, self.P.A[0]) not in self.N:
            for a in self.P.A:
                self.N[(s, a)] = 0
                self.Q[(s, a)] = 0.0
            return self.U(s)
        a = self.explore(s)

        # Call custom_TR with steps to get next state and reward
        s_prime, r = self.P.TR(s, a, steps)

        # Handle invalid transitions directly
        if s_prime is None:
            return r  # Large penalty already returned by custom_TR
        
        # Increment steps for the next simulation call
        q = r + self.P.gamma * self.simulate(s_prime, d - 1, steps + 1)

        # Update visit counts and Q-values
        self.N[(s, a)] += 1
        self.Q[(s, a)] += (q - self.Q[(s, a)]) / self.N[(s, a)]
        return q

    def explore(self, s: Any) -> Any:
        A, N = self.P.A, self.N
        Ns = np.sum([N[(s, a)] for a in A])
        return A[np.argmax([self.ucb1(s, a, Ns) for a in A])]

    def ucb1(self, s: Any, a: Any, Ns: int) -> float:
        N, Q, c = self.N, self.Q, self.c
        return Q[(s, a)] + c*self.bonus(N[(s, a)], Ns)

    @staticmethod
    def bonus(Nsa: int, Ns: int) -> float:
        return np.inf if Nsa == 0 else np.sqrt(np.log(Ns)/Nsa)
