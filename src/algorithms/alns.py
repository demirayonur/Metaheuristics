"""
Adaptive Large Neighbourhood Search (ALNS)
==========================================

A destroy-and-repair metaheuristic with adaptive operator selection.

Algorithm sketch
----------------

1. Start from an initial feasible solution.
2. Select a **destroy** operator (roulette-wheel on adaptive weights).
3. Select a **repair** operator (same mechanism).
4. Destroy: remove a fraction of customers.
5. Repair: re-insert the removed customers.
6. Accept/reject the new solution (SA-style acceptance).
7. Update operator weights based on performance.
8. Repeat until termination.

Destroy operators
-----------------
* **Random removal** — remove customers uniformly at random.
* **Worst removal** — remove the most costly customers.
* **Related removal** — remove a cluster of nearby customers.

Repair operators
----------------
* **Greedy insertion** — insert each customer at the cheapest position.
* **Regret-2 insertion** — prioritise "urgent" customers whose
  second-best insertion is much worse than the best.
"""

from __future__ import annotations

import numpy as np

from ..problems.vrp import VRPData, VRPSolution, VRPObjective


# ════════════════════════════════════════════════════════════════════
# DESTROY OPERATORS
# ════════════════════════════════════════════════════════════════════

class DestroyOperator:
    """
    Base class for ALNS destroy operators.

    Subclasses should override :meth:`destroy`.
    """

    def __init__(self, name: str):
        self.name = name

    def destroy(
        self,
        solution: VRPSolution,
        n_remove: int,
        data: VRPData,
        objective: VRPObjective,
        rng: np.random.Generator,
    ) -> tuple[VRPSolution, list[int]]:
        """
        Remove *n_remove* customers from *solution*.

        Returns
        -------
        partial_solution : VRPSolution
            Solution with customers removed.
        removed : list[int]
            Customer indices that were taken out.
        """
        raise NotImplementedError("Subclasses must implement destroy()")


class RandomDestroy(DestroyOperator):
    """Remove customers uniformly at random."""

    def __init__(self):
        super().__init__("Random Removal")

    def destroy(self, solution, n_remove, data, objective, rng):
        sol = solution.copy()
        customers = list(sol.all_customers())
        n_remove = min(n_remove, len(customers))
        removed = list(rng.choice(customers, n_remove, replace=False))
        for c in removed:
            sol.remove_customer(c)
        sol.remove_empty_routes()
        return sol, removed


class WorstDestroy(DestroyOperator):
    """Remove customers with the highest marginal cost contribution."""

    def __init__(self):
        super().__init__("Worst Removal")

    def destroy(self, solution, n_remove, data, objective, rng):
        sol = solution.copy()
        d = data.dist_matrix

        costs = {}
        for route in sol.routes:
            for idx, cust in enumerate(route):
                prev = 0 if idx == 0 else route[idx - 1]
                nxt = 0 if idx == len(route) - 1 else route[idx + 1]
                cost_with = d[prev, cust] + d[cust, nxt]
                cost_without = d[prev, nxt]
                costs[cust] = cost_with - cost_without

        sorted_custs = sorted(costs, key=costs.get, reverse=True)
        removed = []
        for c in sorted_custs:
            if len(removed) >= n_remove:
                break
            if rng.random() < 0.8:
                removed.append(c)
                sol.remove_customer(c)
        sol.remove_empty_routes()
        return sol, removed


class RelatedDestroy(DestroyOperator):
    """Remove a cluster of geographically close customers."""

    def __init__(self):
        super().__init__("Related Removal")

    def destroy(self, solution, n_remove, data, objective, rng):
        sol = solution.copy()
        customers = list(sol.all_customers())
        if not customers:
            return sol, []

        seed = int(rng.choice(customers))
        dists = sorted(
            [(c, data.dist_matrix[seed, c]) for c in customers],
            key=lambda x: x[1],
        )
        removed = [c for c, _ in dists[:n_remove]]
        for c in removed:
            sol.remove_customer(c)
        sol.remove_empty_routes()
        return sol, removed


# ════════════════════════════════════════════════════════════════════
# REPAIR OPERATORS
# ════════════════════════════════════════════════════════════════════

class RepairOperator:
    """
    Base class for ALNS repair operators.

    Subclasses should override :meth:`repair`.
    """

    def __init__(self, name: str):
        self.name = name

    def repair(
        self,
        solution: VRPSolution,
        removed: list[int],
        data: VRPData,
        objective: VRPObjective,
        rng: np.random.Generator,
    ) -> VRPSolution:
        """Insert all *removed* customers back into *solution*."""
        raise NotImplementedError("Subclasses must implement repair()")


class GreedyRepair(RepairOperator):
    """Insert each customer at the position with minimum insertion cost."""

    def __init__(self):
        super().__init__("Greedy Insertion")

    def repair(self, solution, removed, data, objective, rng):
        sol = solution.copy()
        d = data.dist_matrix
        rng.shuffle(removed)

        for cust in removed:
            best_cost = np.inf
            best_ri, best_pos = -1, -1

            for ri, route in enumerate(sol.routes):
                if (objective.route_load(route) + data.demands[cust]
                        > data.vehicle_capacity):
                    continue
                for pos in range(len(route) + 1):
                    prev = 0 if pos == 0 else route[pos - 1]
                    nxt = 0 if pos == len(route) else route[pos]
                    cost = d[prev, cust] + d[cust, nxt] - d[prev, nxt]
                    if cost < best_cost:
                        best_cost, best_ri, best_pos = cost, ri, pos

            if best_ri == -1:
                sol.routes.append([cust])
            else:
                sol.routes[best_ri].insert(best_pos, cust)
        return sol


class RegretRepair(RepairOperator):
    """
    Regret-2 insertion.

    Prioritise customers whose second-best insertion is much more
    expensive than their best, because they are most "urgent" to
    place well now.
    """

    def __init__(self):
        super().__init__("Regret-2 Insertion")

    def repair(self, solution, removed, data, objective, rng):
        sol = solution.copy()
        d = data.dist_matrix
        remaining = list(removed)

        while remaining:
            best_regret = -np.inf
            best_cust, best_ri, best_pos = None, -1, -1

            for cust in remaining:
                ins_costs = []
                for ri, route in enumerate(sol.routes):
                    if (objective.route_load(route) + data.demands[cust]
                            > data.vehicle_capacity):
                        continue
                    for pos in range(len(route) + 1):
                        prev = 0 if pos == 0 else route[pos - 1]
                        nxt = 0 if pos == len(route) else route[pos]
                        cost = d[prev, cust] + d[cust, nxt] - d[prev, nxt]
                        ins_costs.append((cost, ri, pos))

                new_route_cost = d[0, cust] + d[cust, 0]
                ins_costs.append((new_route_cost, -1, 0))
                ins_costs.sort(key=lambda x: x[0])

                c1 = ins_costs[0][0]
                c2 = ins_costs[1][0] if len(ins_costs) > 1 else c1
                regret = c2 - c1

                if regret > best_regret:
                    best_regret = regret
                    best_cust = cust
                    best_ri = ins_costs[0][1]
                    best_pos = ins_costs[0][2]

            remaining.remove(best_cust)
            if best_ri == -1:
                sol.routes.append([best_cust])
            else:
                sol.routes[best_ri].insert(best_pos, best_cust)

        return sol


# ════════════════════════════════════════════════════════════════════
# ALNS ENGINE
# ════════════════════════════════════════════════════════════════════

class ALNS:
    """
    Adaptive Large Neighbourhood Search for CVRP.

    Parameters
    ----------
    data : VRPData
        Problem instance.
    destroy_operators : list[DestroyOperator] or None
        Custom destroy operators (defaults to Random + Worst + Related).
    repair_operators : list[RepairOperator] or None
        Custom repair operators (defaults to Greedy + Regret-2).
    max_iterations : int
        Total number of ALNS iterations.
    segment_size : int
        Iterations per adaptive-weight-update segment.
    removal_fraction : tuple[float, float]
        ``(min_frac, max_frac)`` — bounds on fraction of customers
        to remove per destroy step.
    sigma1, sigma2, sigma3 : float
        Score rewards for: *new global best*, *improving*, *accepted worse*.
    decay : float
        Weight smoothing factor (0-1).
    initial_temp : float
        SA acceptance starting temperature.
    cooling_rate : float
        SA cooling factor per iteration.

    Attributes (populated after :meth:`run`)
    -----------------------------------------
    history_best : list[float]
    history_current : list[float]
    history_temps : list[float]
    d_weight_history : list[numpy.ndarray]
    r_weight_history : list[numpy.ndarray]
    """

    def __init__(
        self,
        data: VRPData,
        destroy_operators: list[DestroyOperator] | None = None,
        repair_operators: list[RepairOperator] | None = None,
        max_iterations: int = 3000,
        segment_size: int = 100,
        removal_fraction: tuple[float, float] = (0.15, 0.40),
        sigma1: float = 33.0,
        sigma2: float = 9.0,
        sigma3: float = 13.0,
        decay: float = 0.8,
        initial_temp: float = 50.0,
        cooling_rate: float = 0.9995,
    ):
        self.data = data
        self.objective = VRPObjective(data)

        self.destroy_ops = destroy_operators or [
            RandomDestroy(), WorstDestroy(), RelatedDestroy(),
        ]
        self.repair_ops = repair_operators or [
            GreedyRepair(), RegretRepair(),
        ]

        self.max_iterations = max_iterations
        self.segment_size = segment_size
        self.removal_fraction = removal_fraction
        self.sigma1, self.sigma2, self.sigma3 = sigma1, sigma2, sigma3
        self.decay = decay
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate

        # Adaptive weights
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))

        # History
        self.history_best: list[float] = []
        self.history_current: list[float] = []
        self.history_temps: list[float] = []
        self.d_weight_history: list[np.ndarray] = []
        self.r_weight_history: list[np.ndarray] = []

    def _select_operator(
        self, weights: np.ndarray, rng: np.random.Generator
    ) -> int:
        """Roulette-wheel selection proportional to weights."""
        probs = weights / weights.sum()
        return int(rng.choice(len(weights), p=probs))

    def _initial_solution(self, rng: np.random.Generator) -> VRPSolution:
        """Build an initial feasible solution with greedy bin-packing."""
        customers = list(range(1, self.data.n_customers + 1))
        rng.shuffle(customers)
        routes: list[list[int]] = []
        for c in customers:
            placed = False
            for route in routes:
                load = sum(self.data.demands[x] for x in route)
                if load + self.data.demands[c] <= self.data.vehicle_capacity:
                    route.append(c)
                    placed = True
                    break
            if not placed:
                routes.append([c])
        return VRPSolution(routes)

    def run(self, seed: int = 42) -> tuple[VRPSolution, float]:
        """
        Execute the ALNS algorithm.

        Parameters
        ----------
        seed : int
            Random seed.

        Returns
        -------
        best_solution : VRPSolution
        best_distance : float
        """
        rng = np.random.default_rng(seed)

        current = self._initial_solution(rng)
        current_cost = self.objective.evaluate(current)
        best = current.copy()
        best_cost = current_cost
        temp = self.initial_temp

        n_custs = self.data.n_customers
        min_rem = max(1, int(self.removal_fraction[0] * n_custs))
        max_rem = max(2, int(self.removal_fraction[1] * n_custs))

        d_scores = np.zeros(len(self.destroy_ops))
        r_scores = np.zeros(len(self.repair_ops))
        d_counts = np.zeros(len(self.destroy_ops))
        r_counts = np.zeros(len(self.repair_ops))

        self.history_best = []
        self.history_current = []
        self.history_temps = []
        self.d_weight_history = []
        self.r_weight_history = []

        for it in range(1, self.max_iterations + 1):
            di = self._select_operator(self.d_weights, rng)
            ri = self._select_operator(self.r_weights, rng)
            n_remove = rng.integers(min_rem, max_rem + 1)

            partial, removed = self.destroy_ops[di].destroy(
                current, n_remove, self.data, self.objective, rng
            )
            candidate = self.repair_ops[ri].repair(
                partial, removed, self.data, self.objective, rng
            )
            candidate_cost = self.objective.evaluate(candidate)

            d_counts[di] += 1
            r_counts[ri] += 1

            if candidate_cost < best_cost:
                best, best_cost = candidate.copy(), candidate_cost
                current, current_cost = candidate, candidate_cost
                d_scores[di] += self.sigma1
                r_scores[ri] += self.sigma1
            elif candidate_cost < current_cost:
                current, current_cost = candidate, candidate_cost
                d_scores[di] += self.sigma2
                r_scores[ri] += self.sigma2
            else:
                delta = candidate_cost - current_cost
                if temp > 0 and rng.random() < np.exp(-delta / temp):
                    current, current_cost = candidate, candidate_cost
                    d_scores[di] += self.sigma3
                    r_scores[ri] += self.sigma3

            temp *= self.cooling_rate

            self.history_best.append(best_cost)
            self.history_current.append(current_cost)
            self.history_temps.append(temp)

            # Update weights
            if it % self.segment_size == 0:
                for k in range(len(self.destroy_ops)):
                    if d_counts[k] > 0:
                        self.d_weights[k] = (
                            self.decay * self.d_weights[k]
                            + (1 - self.decay) * d_scores[k] / d_counts[k]
                        )
                for k in range(len(self.repair_ops)):
                    if r_counts[k] > 0:
                        self.r_weights[k] = (
                            self.decay * self.r_weights[k]
                            + (1 - self.decay) * r_scores[k] / r_counts[k]
                        )
                self.d_weight_history.append(self.d_weights.copy())
                self.r_weight_history.append(self.r_weights.copy())
                d_scores[:] = 0
                r_scores[:] = 0
                d_counts[:] = 0
                r_counts[:] = 0

        return best, best_cost
