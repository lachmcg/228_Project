import numpy as np

from MDP import MDP
from abc import abstractmethod, ABC
from typing import Any, Callable

class ValueFunctionPolicy():
    def __init__(self, P: MDP, U: Callable[[Any], float] | np.ndarray):
        self.P = P  # problem
        self.U = U  # utility function

    def __call__(self, s: Any) -> Any:
        return self.P.greedy(self.U, s)[0]


class MDPSolutionMethod(ABC):
    pass


class OfflinePlanningMethod(MDPSolutionMethod):
    @abstractmethod
    def solve(self, P: MDP) -> Callable[[Any], Any]:
        pass


class ExactSolutionMethod(OfflinePlanningMethod):
    pass


class PolicyIteration(ExactSolutionMethod):
    def __init__(self, initial_policy: Callable[[Any], Any], k_max: int):
        self.initial_policy = initial_policy
        self.k_max = k_max

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        policy = self.initial_policy
        for _ in range(self.k_max):
            U = P.policy_evaluation(policy)
            policy_prime = ValueFunctionPolicy(P, U)
            if all([policy(s) == policy_prime(s) for s in P.S]):
                break
            policy = policy_prime
        return policy


class ValueIteration(ExactSolutionMethod):
    def __init__(self, k_max: int):
        self.k_max = k_max

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        U = np.zeros(len(P.S))
        for _ in range(self.k_max):
            U = np.array([P.backup(U, s) for s in P.S])
        return ValueFunctionPolicy(P, U)


class GaussSeidelValueIteration(ExactSolutionMethod):
    def __init__(self, k_max: int):
        self.k_max = k_max

    def solve(self, P: MDP) -> Callable[[Any], Any]:
        U = np.zeros(len(P.S))
        for _ in range(self.k_max):
            for i, s in enumerate(P.S):
                U[i] = P.backup(U, s)
        return ValueFunctionPolicy(P, U)


class LinearProgramFormulation(ExactSolutionMethod):
    def solve(self, P: MDP) -> Callable[[Any], Any]:
        S, A, R, T = self.numpyform(P)
        U = cp.Variable(len(S))
        objective = cp.Minimize(cp.sum(U))
        constraints = [U[s] >= R[s, a] + P.gamma * (T[s, a] @ U) for s in S for a in A]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return ValueFunctionPolicy(P, U.value)

    @staticmethod
    def numpyform(P: MDP) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        S_prime = np.arange(len(P.S))
        A_prime = np.arange(len(P.A))
        R_prime = np.array([[P.R(s, a) for a in P.A] for s in P.S])
        T_prime = np.array([[[P.T(s, a, s_prime) for s_prime in S_prime] for a in P.A] for s in P.S])
        return S_prime, A_prime, R_prime, T_prime
