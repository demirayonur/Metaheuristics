"""
Genetic Algorithm (GA) for TSP
==============================

A population-based metaheuristic inspired by biological evolution.

Algorithm sketch
----------------

1. Initialise a population of random tours.
2. **Select** parents via tournament selection.
3. **Crossover** — Order Crossover (OX) produces offspring.
4. **Mutate** — swap two random cities.
5. **Local search** — apply a few 2-opt improvement moves.
6. **Survive** — generational replacement with elitism.
7. Repeat for *G* generations.
"""

from __future__ import annotations

import numpy as np

from ..problems.tsp import TSPData, TSPSolution, TSPObjective


class GeneticAlgorithmTSP:
    """
    Genetic Algorithm for the Travelling Salesman Problem.

    Parameters
    ----------
    data : TSPData
        Problem instance.
    pop_size : int
        Population size.
    n_generations : int
        Number of generations to evolve.
    crossover_rate : float
        Probability of applying crossover to a pair of parents.
    mutation_rate : float
        Probability of mutating an offspring.
    tournament_size : int
        Number of individuals competing in tournament selection.
    elite_count : int
        Number of best individuals carried over unchanged.
    local_search_iters : int
        Number of random 2-opt moves per offspring.

    Attributes (populated after :meth:`run`)
    -----------------------------------------
    best_per_gen : list[float]
        Best fitness in each generation.
    avg_per_gen : list[float]
        Mean fitness in each generation.
    worst_per_gen : list[float]
        Worst fitness in each generation.
    diversity_per_gen : list[float]
        Population diversity metric per generation.
    """

    def __init__(
        self,
        data: TSPData,
        pop_size: int = 100,
        n_generations: int = 500,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.3,
        tournament_size: int = 5,
        elite_count: int = 2,
        local_search_iters: int = 20,
    ):
        self.data = data
        self.objective = TSPObjective(data)
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.local_search_iters = local_search_iters

        self.best_per_gen: list[float] = []
        self.avg_per_gen: list[float] = []
        self.worst_per_gen: list[float] = []
        self.diversity_per_gen: list[float] = []

    # ── operators ───────────────────────────────────────────────────

    def _random_tour(self, rng: np.random.Generator) -> TSPSolution:
        """Generate a random permutation tour."""
        return TSPSolution(list(rng.permutation(self.data.n_cities)))

    def _tournament_select(
        self,
        population: list[TSPSolution],
        fitnesses: list[float],
        rng: np.random.Generator,
    ) -> TSPSolution:
        """Select one individual via tournament selection."""
        idxs = rng.choice(len(population), size=self.tournament_size,
                          replace=False)
        best_idx = min(idxs, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _order_crossover(
        self, p1: TSPSolution, p2: TSPSolution, rng: np.random.Generator
    ) -> TSPSolution:
        """
        Order Crossover (OX).

        1. Copy a random substring from *p1* into the child.
        2. Fill remaining positions with cities from *p2* in order,
           skipping cities already present.
        """
        n = len(p1.tour)
        start, end = sorted(rng.choice(n, 2, replace=False))
        child = [-1] * n
        child[start : end + 1] = p1.tour[start : end + 1]
        used = set(child[start : end + 1])

        pos = (end + 1) % n
        for city in p2.tour:
            if city not in used:
                child[pos] = city
                pos = (pos + 1) % n
        return TSPSolution(child)

    def _swap_mutation(
        self, sol: TSPSolution, rng: np.random.Generator
    ) -> TSPSolution:
        """Swap two random cities in the tour."""
        s = sol.copy()
        i, j = rng.choice(len(s.tour), 2, replace=False)
        s.tour[i], s.tour[j] = s.tour[j], s.tour[i]
        return s

    def _two_opt_improve(
        self, sol: TSPSolution, rng: np.random.Generator
    ) -> TSPSolution:
        """Apply a few random 2-opt moves, keeping improvements."""
        s = sol.copy()
        d = self.data.dist_matrix
        n = len(s.tour)
        for _ in range(self.local_search_iters):
            i, j = sorted(rng.choice(n, 2, replace=False))
            if j - i < 2:
                continue
            a, b = s.tour[i], s.tour[(i + 1) % n]
            c, e = s.tour[j], s.tour[(j + 1) % n]
            if d[a, b] + d[c, e] > d[a, c] + d[b, e]:
                s.tour[i + 1 : j + 1] = reversed(s.tour[i + 1 : j + 1])
        return s

    def _population_diversity(
        self, population: list[TSPSolution]
    ) -> float:
        """Average pairwise edge-set difference (sampled)."""
        n_sample = min(20, len(population))
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(population), n_sample, replace=False)

        def edge_set(tour):
            return {
                (min(tour[k], tour[(k + 1) % len(tour)]),
                 max(tour[k], tour[(k + 1) % len(tour)]))
                for k in range(len(tour))
            }

        diffs = []
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                e1 = edge_set(population[idxs[i]].tour)
                e2 = edge_set(population[idxs[j]].tour)
                diffs.append(
                    len(e1.symmetric_difference(e2)) / max(len(e1), 1)
                )
        return float(np.mean(diffs)) if diffs else 0.0

    # ── main loop ───────────────────────────────────────────────────

    def run(self, seed: int = 42) -> tuple[TSPSolution, float]:
        """
        Execute the Genetic Algorithm.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        best_solution : TSPSolution
        best_distance : float
        """
        rng = np.random.default_rng(seed)

        # Initialise
        population = [self._random_tour(rng) for _ in range(self.pop_size)]
        fitnesses = [self.objective.evaluate(ind) for ind in population]

        best_idx = int(np.argmin(fitnesses))
        global_best = population[best_idx].copy()
        global_best_fit = fitnesses[best_idx]

        self.best_per_gen = []
        self.avg_per_gen = []
        self.worst_per_gen = []
        self.diversity_per_gen = []

        for gen in range(self.n_generations):
            # Record stats
            self.best_per_gen.append(min(fitnesses))
            self.avg_per_gen.append(float(np.mean(fitnesses)))
            self.worst_per_gen.append(max(fitnesses))
            if gen % 25 == 0:
                self.diversity_per_gen.append(
                    self._population_diversity(population)
                )
            else:
                self.diversity_per_gen.append(
                    self.diversity_per_gen[-1]
                    if self.diversity_per_gen
                    else 1.0
                )

            # Elitism
            order = np.argsort(fitnesses)
            new_pop = [
                population[order[i]].copy() for i in range(self.elite_count)
            ]

            # Fill rest
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)

                child = (
                    self._order_crossover(p1, p2, rng)
                    if rng.random() < self.crossover_rate
                    else p1.copy()
                )

                if rng.random() < self.mutation_rate:
                    child = self._swap_mutation(child, rng)

                child = self._two_opt_improve(child, rng)
                new_pop.append(child)

            population = new_pop[: self.pop_size]
            fitnesses = [self.objective.evaluate(ind) for ind in population]

            gen_best_idx = int(np.argmin(fitnesses))
            if fitnesses[gen_best_idx] < global_best_fit:
                global_best = population[gen_best_idx].copy()
                global_best_fit = fitnesses[gen_best_idx]

        return global_best, global_best_fit
