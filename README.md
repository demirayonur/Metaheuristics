# Metaheuristics: From Local Search to Adaptive Large Neighbourhood Search

> **A hands-on lecture series with Python implementations and exact-solver benchmarks.**

This repository contains a complete, self-contained tutorial on metaheuristic optimisation.
Every algorithm is implemented from scratch in object-oriented Python, benchmarked against
[Gurobi](https://www.gurobi.com/) exact solutions, and accompanied by publication-quality
figures that you can use directly in your slides.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Structure](#repository-structure)
3. [Module 1 — What Is Local Search?](#module-1--what-is-local-search)
4. [Module 2 — Why We Need Metaheuristics](#module-2--why-we-need-metaheuristics)
5. [Module 3 — Anatomy of a Metaheuristic](#module-3--anatomy-of-a-metaheuristic)
6. [Module 4 — Simulated Annealing (Knapsack)](#module-4--simulated-annealing)
7. [Module 5 — Genetic Algorithm (TSP)](#module-5--genetic-algorithm)
8. [Module 6 — Adaptive Large Neighbourhood Search (VRP)](#module-6--adaptive-large-neighbourhood-search)
9. [Generating Figures](#generating-figures)
10. [Extending the Framework](#extending-the-framework)
11. [License](#license)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/metaheuristics-tutorial.git
cd metaheuristics-tutorial

# 2. Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run everything and generate all figures
python -m examples.run_all
```

**Prerequisites:**

| Dependency | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | f-strings, `match`, type unions |
| NumPy | ≥ 1.24 | Array operations, RNG |
| Matplotlib | ≥ 3.7 | Visualisation |
| Gurobi | ≥ 10.0 | Exact MIP/LP benchmarks |

> **Note on Gurobi:** A free academic licence or the restricted
> licence (ships with `pip install gurobipy`) is sufficient for
> the instance sizes used here.

---

## Repository Structure

```
metaheuristics-tutorial/
│
├── src/
│   ├── core/                  # Abstract building blocks (Module 3)
│   │   ├── solution.py        #   Solution base class
│   │   ├── objective.py       #   ObjectiveFunction base class
│   │   ├── neighbourhood.py   #   Neighbourhood base class
│   │   ├── acceptance.py      #   Greedy & Metropolis acceptance
│   │   ├── termination.py     #   Composite stopping criteria
│   │   ├── history.py         #   Search trajectory recorder
│   │   └── metaheuristic.py   #   Abstract metaheuristic skeleton
│   │
│   ├── problems/              # Problem definitions
│   │   ├── knapsack.py        #   0/1 Knapsack: data, solution, objective, neighbourhood
│   │   ├── tsp.py             #   TSP: data, solution, objective
│   │   └── vrp.py             #   CVRP: data, solution, objective
│   │
│   ├── algorithms/            # Metaheuristic implementations
│   │   ├── simulated_annealing.py   # SA with geometric cooling
│   │   ├── genetic_algorithm.py     # GA with OX crossover + 2-opt
│   │   └── alns.py                  # ALNS with 3 destroy + 2 repair operators
│   │
│   ├── solvers/               # Gurobi exact solvers
│   │   ├── gurobi_knapsack.py
│   │   ├── gurobi_tsp.py       # MTZ formulation
│   │   └── gurobi_vrp.py       # Two-index vehicle-flow + MTZ
│   │
│   └── visualization/         # Plotting functions
│       ├── style.py            #   Shared palette & rcParams
│       ├── plot_local_search.py
│       ├── plot_components.py
│       ├── plot_sa.py
│       ├── plot_ga.py
│       └── plot_alns.py
│
├── examples/                  # Runnable scripts
│   ├── run_all.py             #   Run everything at once
│   ├── run_local_search.py    #   Modules 1 & 2 demo
│   ├── run_sa_knapsack.py     #   Module 4 demo
│   ├── run_ga_tsp.py          #   Module 5 demo
│   └── run_alns_vrp.py        #   Module 6 demo
│
├── docs/figures/              # Generated plots (after running examples)
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md                  # ← You are here
```

---

## Module 1 — What Is Local Search?

### The Idea

Local search is the simplest optimisation strategy:

1. **Start** from some initial solution *s*.
2. **Look around** — examine the *neighbourhood* N(s), the set of solutions reachable in one step.
3. **Move** to the best improving neighbour.
4. **Repeat** until no neighbour is better than the current solution.

When you stop, you are at a **local optimum** — a solution that is at least as good as
everything "nearby", but not necessarily the best solution overall.

### Key Vocabulary

| Term | Meaning |
|---|---|
| **Solution representation** | The data structure encoding a candidate answer (binary vector, permutation, route list, …) |
| **Neighbourhood** | The set of solutions reachable from the current one in a single move |
| **Move operator** | The function that transforms one solution into a neighbour (flip a bit, swap two cities, …) |
| **Objective function** | A scalar measure of solution quality — we minimise throughout this tutorial |
| **Local optimum** | A solution with no improving neighbour — the search *gets stuck* here |

### Demo

```bash
python -m examples.run_local_search
```

We demonstrate on the 1-D **Rastrigin function**, a classic test landscape with many
local minima.  Starting from x₀ = 3.5, hill-climbing converges to a local minimum at
x = 3.0 with f(x) = 9.0 — far from the global minimum at x = 0, f(x) = 0.

![Module 1 & 2](docs/figures/module1_2_local_search.png)

---

## Module 2 — Why We Need Metaheuristics

### The Problem with Greedy

Local search is *greedy*: it only accepts improvements.  On a landscape with many local
optima, this means the search **gets trapped** — the final result depends entirely on
where you start, and most starting points lead to mediocre solutions.

### First Remedy: Multi-Start

The simplest escape strategy is to run local search many times from different random
starting points and keep the overall best.  With 30 restarts on the Rastrigin function,
multi-start finds f ≈ 0.017 — much closer to the global optimum.

### Beyond Multi-Start

Multi-start is wasteful: each restart throws away everything the previous search learned.
**Metaheuristics** are smarter — they use various strategies to escape local optima
*during* a single continuous search:

| Strategy | Metaheuristic | How it escapes |
|---|---|---|
| Accept worse moves probabilistically | **Simulated Annealing** | Temperature-controlled random uphill steps |
| Maintain a diverse population | **Genetic Algorithm** | Crossover combines good building blocks |
| Destroy and rebuild parts of the solution | **ALNS** | Large moves jump to distant regions |
| Remember and forbid recent moves | **Tabu Search** | Short-term memory prevents cycling |

---

## Module 3 — Anatomy of a Metaheuristic

### The Shared Skeleton

Despite their differences, nearly all metaheuristics share **six core components**.
Our codebase in `src/core/` implements each one as an abstract base class:

![Module 3 — Components](docs/figures/module3_components.png)

#### 1. Solution Representation (`solution.py`)

How we encode a candidate answer.  This is problem-specific:

```python
# Binary vector for Knapsack
class KnapsackSolution(Solution):
    selection: np.ndarray          # e.g. [1, 0, 1, 1, 0, ...]

# Permutation for TSP
class TSPSolution(Solution):
    tour: list[int]                # e.g. [0, 4, 2, 7, 1, ...]

# List of routes for VRP
class VRPSolution(Solution):
    routes: list[list[int]]        # e.g. [[3, 7], [1, 5, 2]]
```

#### 2. Objective Function (`objective.py`)

Measures solution quality.  **Convention: we always minimise.**
For maximisation problems (like Knapsack profit), negate the value.

```python
class KnapsackObjective(ObjectiveFunction):
    def evaluate(self, solution) -> float:
        return -total_value + penalty * max(0, total_weight - capacity)
```

#### 3. Neighbourhood Definition (`neighbourhood.py`)

Defines which solutions are "one step away".  This is the **most important design
decision** — it controls the landscape the search explores.

```python
class KnapsackFlipNeighbourhood(Neighbourhood):
    """Flip one bit — pack or unpack one item."""

    def get_random_neighbour(self, solution, rng):
        neighbour = solution.copy()
        idx = rng.integers(0, self.n_items)
        neighbour.selection[idx] = 1 - neighbour.selection[idx]
        return neighbour
```

#### 4. Initial Solution Construction

Where does the search start?  Options range from purely random to sophisticated greedy
heuristics.  Better starting points usually help, but diversity matters too.

#### 5. Acceptance Criterion (`acceptance.py`)

The critical decision: do we move to the neighbour, or stay put?

```python
class GreedyAcceptance:       # Only improvements — this IS local search
    accept = candidate_cost < current_cost

class MetropolisAcceptance:   # SA-style — sometimes accept worse
    accept = (delta < 0) or (random() < exp(-delta / T))
```

#### 6. Termination Criteria (`termination.py`)

When do we stop?  Our `TerminationCriteria` class bundles four independent conditions —
any one being met triggers termination:

- Maximum iterations reached
- Wall-clock time limit exceeded
- No improvement for *k* consecutive iterations
- Target objective value achieved

### The Run Loop (`metaheuristic.py`)

```python
class Metaheuristic(ABC):
    def run(self, seed=42):
        solution = self._build_initial(rng)            # Hook 1
        for iteration in range(max_iter):
            self._iterate(iteration, rng)              # Hook 2
            if termination.should_stop(...):
                break
        return best_solution, best_cost, history
```

Subclasses only implement two methods: `_build_initial()` and `_iterate()`.
Everything else — timing, history recording, termination checks — is handled
by the base class.

---

## Module 4 — Simulated Annealing

### Theory

**Simulated Annealing** (Kirkpatrick et al., 1983) borrows the idea of controlled
cooling from metallurgy.  The key insight: at high temperatures, the search freely
explores (accepts many worse solutions); as the temperature drops, it becomes
increasingly selective and converges to a good solution.

**The Metropolis criterion:**

- If the neighbour is **better** → always accept.
- If the neighbour is **worse** by Δf → accept with probability:

$$P(\text{accept}) = \exp\!\left(\frac{-\Delta f}{T}\right)$$

As temperature *T* → 0, this probability → 0, and SA degenerates into greedy local search.

**Cooling schedule (geometric):**

$$T_{k+1} = \alpha \cdot T_k, \quad \alpha \in [0.99, 0.9999]$$

### Application: 0/1 Knapsack (50 items)

| Component | Choice |
|---|---|
| Representation | Binary vector `x ∈ {0,1}^50` |
| Objective | `-Σ vᵢxᵢ + ρ · max(0, Σ wᵢxᵢ − C)` |
| Neighbourhood | Single bit flip |
| Initial solution | Greedy (sort by value/weight ratio) |
| Acceptance | Metropolis |
| Cooling | α = 0.9985, T₀ = 100, T_min = 0.001 |
| Reheating | Partial reheat every 1500 non-improving iterations |
| Termination | 10,000 iterations or 2,000 without improvement |

### Run

```bash
python -m examples.run_sa_knapsack
```

### Results

```
  [Gurobi] optimal value = 1966
  [SA]     value          = 1966
  >>> Optimality gap      = 0.00%
```

![Module 4 — SA](docs/figures/module4_sa_knapsack.png)

**Panel (a)** shows the SA convergence: the blue trace (current profit) oscillates
wildly at first (high temperature → many worse moves accepted) and stabilises as the
system cools.  **Panel (b)** shows the geometric cooling schedule on a log scale.
**Panel (c)** confirms that SA matches Gurobi's exact optimum.  **Panel (d)** compares
item-by-item selection — green = selected, red = not selected.

---

## Module 5 — Genetic Algorithm

### Theory

**Genetic Algorithms** (Holland, 1975) maintain a *population* of solutions and evolve
them through operators inspired by biological evolution:

1. **Selection** — choose parents, biased toward fitter individuals.
   We use **tournament selection** (pick *k* random individuals, keep the best).

2. **Crossover** — combine two parents to produce offspring.
   For permutations (TSP), we use **Order Crossover (OX)**:
   - Copy a random substring from Parent 1.
   - Fill remaining positions with cities from Parent 2, in order, skipping duplicates.

3. **Mutation** — introduce small random changes (swap two cities).

4. **Local search** — hybrid GAs apply a few iterations of local improvement (2-opt)
   to each offspring.  This is sometimes called a **memetic algorithm**.

5. **Survival** — the next generation is formed.  **Elitism** carries the best
   individuals forward unchanged.

**Diversity** is crucial: if the population converges too quickly, the GA gets stuck.
We track diversity using the average pairwise edge-set difference between tours.

### Application: TSP (15 cities)

| Component | Choice |
|---|---|
| Representation | Permutation of city indices |
| Objective | Total Euclidean tour length |
| Population | 80 individuals |
| Selection | Tournament (k = 5) |
| Crossover | Order Crossover (OX), rate = 0.9 |
| Mutation | Swap, rate = 0.35 |
| Local search | 30 random 2-opt moves per offspring |
| Elitism | Top 2 carried over |
| Generations | 300 |

### Run

```bash
python -m examples.run_ga_tsp
```

### Results

```
  [Gurobi] optimal distance = 313.00
  [GA]     distance          = 313.00
  >>> Optimality gap         = 0.00%
```

![Module 5 — GA](docs/figures/module5_ga_tsp.png)

**Panel (a–b)** compare the GA's best tour to Gurobi's optimal tour visually.
**Panel (c)** shows convergence: the shaded band spans worst-to-best in the population,
and the red line tracks the best individual.  **Panel (d)** tracks population diversity —
note how it drops as the population converges.  **Panel (f)** highlights shared edges
(green) vs. edges unique to one solution.

### Gurobi Formulation: Miller–Tucker–Zemlin (MTZ)

The exact TSP solver uses the MTZ subtour-elimination formulation:

$$\min \sum_{i \neq j} d_{ij}\, x_{ij}$$

Subject to:
- Each city entered and left exactly once.
- Subtour elimination: $u_i - u_j + n \cdot x_{ij} \leq n - 1$ for $i, j \geq 1$.

---

## Module 6 — Adaptive Large Neighbourhood Search

### Theory

**ALNS** (Ropke & Pisinger, 2006) is a destroy-and-repair framework that maintains
a portfolio of operators and **adaptively** learns which ones work best on the
current instance.

**The ALNS loop:**

1. **Select** a destroy operator and a repair operator using **roulette-wheel
   selection** — operators with higher weights are chosen more often.
2. **Destroy** — remove a fraction of the solution (e.g. remove 20–40% of customers).
3. **Repair** — re-insert the removed elements using a constructive heuristic.
4. **Accept/reject** — use an SA-style acceptance criterion.
5. **Score** the operators based on what happened:
   - σ₁ = 33 points if we found a new global best.
   - σ₂ = 9 points if the solution improved.
   - σ₃ = 13 points if a worse solution was accepted (exploration).
6. **Update weights** every *segment* of iterations using exponential smoothing:

$$w_k^{\text{new}} = \lambda \cdot w_k^{\text{old}} + (1 - \lambda) \cdot \frac{\text{score}_k}{\text{count}_k}$$

### Destroy Operators

| Operator | Strategy |
|---|---|
| **Random Removal** | Remove customers uniformly at random |
| **Worst Removal** | Remove the customers that contribute the most to total cost |
| **Related Removal** | Remove a cluster of geographically close customers |

### Repair Operators

| Operator | Strategy |
|---|---|
| **Greedy Insertion** | Insert each customer at the cheapest feasible position |
| **Regret-2 Insertion** | Prioritise customers whose 2nd-best insertion is much worse than the best |

### Application: CVRP (12 customers)

| Component | Choice |
|---|---|
| Representation | List of routes (depot implicit) |
| Objective | Total distance over all routes |
| Destroy operators | Random, Worst, Related |
| Repair operators | Greedy, Regret-2 |
| Removal fraction | 20–40% of customers per iteration |
| Acceptance | SA (T₀ = 50, α = 0.9993) |
| Weight update | Every 100 iterations, λ = 0.8 |
| Iterations | 3,000 |

### Run

```bash
python -m examples.run_alns_vrp
```

### Results

```
  [Gurobi] optimal distance = 502.88  (4 vehicles)
  [ALNS]   distance          = 502.88  (4 vehicles)
  >>> Optimality gap         = 0.00%
```

![Module 6 — ALNS](docs/figures/module6_alns_vrp.png)

**Panel (a–b)** show the ALNS and Gurobi route maps side by side — each colour is a
different vehicle.  **Panel (c)** shows convergence.  **Panel (d)** is unique to ALNS:
it shows how the adaptive weights evolve over time — you can see the algorithm learning
which operators are most effective.  **Panel (f)** shows the SA acceptance temperature
decaying over iterations.

### Gurobi Formulation: Two-Index Vehicle Flow + MTZ

The exact CVRP solver uses:
- Binary arc variables $x_{ij}$.
- Continuous load variables $u_i$ for MTZ subtour/capacity elimination.
- Vehicle count bounded at the depot.

---

## Generating Figures

All figures are generated automatically when you run the examples:

```bash
# Generate ALL figures at once
python -m examples.run_all

# Or generate individually
python -m examples.run_local_search      # → docs/figures/module1_2_local_search.png
python -m examples.run_sa_knapsack       # → docs/figures/module4_sa_knapsack.png
python -m examples.run_ga_tsp            # → docs/figures/module5_ga_tsp.png
python -m examples.run_alns_vrp          # → docs/figures/module6_alns_vrp.png
```

Figures are saved at 180 DPI in `docs/figures/` — suitable for both slides and print.

### Customising Plots

The shared style is defined in `src/visualization/style.py`:

```python
from src.visualization.style import COLORS, apply_style

apply_style()                  # Sets font, DPI, title style
print(COLORS["gurobi"])        # → "#2ecc71"
```

---

## Extending the Framework

### Adding a New Problem

1. Create a file in `src/problems/` with:
   - A `@dataclass` for the problem data.
   - A `Solution` subclass with `copy()` and `__repr__()`.
   - An `ObjectiveFunction` subclass with `evaluate()`.
   - (Optional) a `Neighbourhood` subclass.

2. Register it in `src/problems/__init__.py`.

### Adding a New Algorithm

1. For single-solution methods: subclass `Metaheuristic` from `src/core/`:
   ```python
   class TabuSearch(Metaheuristic):
       def _build_initial(self, rng): ...
       def _iterate(self, iteration, rng): ...
   ```

2. For population-based methods: follow the `GeneticAlgorithmTSP` pattern — no
   inheritance from `Metaheuristic` needed, just implement a `run()` method.

3. Add it to `src/algorithms/__init__.py`.

### Adding a New Gurobi Solver

1. Create a function in `src/solvers/` that takes problem data and returns
   `(optimal_value, solution)`.

2. Register it in `src/solvers/__init__.py`.

### Adding ALNS Operators

```python
from src.algorithms.alns import DestroyOperator, RepairOperator

class ShawDestroy(DestroyOperator):
    """Remove customers that are similar (distance + demand + time)."""
    def __init__(self):
        super().__init__("Shaw Removal")

    def destroy(self, solution, n_remove, data, objective, rng):
        # Your implementation here
        ...

# Plug it in:
alns = ALNS(
    data=data,
    destroy_operators=[RandomDestroy(), WorstDestroy(), ShawDestroy()],
    repair_operators=[GreedyRepair(), RegretRepair()],
)
```

---

## Summary of Results

| Module | Problem | Algorithm | Gurobi Optimal | Metaheuristic | Gap |
|---|---|---|---|---|---|
| 4 | 0/1 Knapsack (50 items) | Simulated Annealing | 1,966 | 1,966 | **0.00%** |
| 5 | TSP (15 cities) | Genetic Algorithm | 313.00 | 313.00 | **0.00%** |
| 6 | CVRP (12 customers) | ALNS | 502.88 | 502.88 | **0.00%** |

> All three metaheuristics find the **proven optimal** solution on these instances.
> On larger instances (where Gurobi may time out or run out of memory),
> metaheuristics continue to produce high-quality solutions in seconds.

---

## References

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). *Optimization by Simulated Annealing.* Science, 220(4598), 671–680.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.* University of Michigan Press.
- Ropke, S., & Pisinger, D. (2006). *An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows.* Transportation Science, 40(4), 455–472.
- Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). *Integer Programming Formulation of Traveling Salesman Problems.* Journal of the ACM, 7(4), 326–329.

---

## License

MIT — see [LICENSE](LICENSE).
