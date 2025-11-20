import numpy as np
import pulp
import time

from TSA import TSA
from problem_generator import generate_problem


def PuTPS(problem_matrix):
    N = problem_matrix.shape[0] - 2

    s = problem_matrix[2:, 0]
    d = problem_matrix[0, 2:]
    c_matrix = problem_matrix[2:, 2:]

    c_matrix[np.isinf(c_matrix)] = 10 ** 6

    model = pulp.LpProblem("PuLP TP solver", pulp.LpMinimize)
    cost = pulp.LpVariable.dicts("cost", (range(N), range(N)), lowBound=0)

    model += pulp.lpSum(c_matrix[i, j] * cost[i][j] for i in range(N) for j in range(N))

    for i in range(N):
        model += pulp.lpSum(cost[i][j] for j in range(N)) == s[i]
        model += pulp.lpSum(cost[j][i] for j in range(N)) == d[i]

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model.objective.value()


problem_sizes = [5, 10, 15, 30]
num_runs = 100

print("=" * 100)
print("TRANSPORTATION SIMPLEX ALGORITHM - PERFORMANCE ANALYSIS")
print("=" * 100)
print(f"\nProblem sizes: {problem_sizes}")
print(f"Runs per size: {num_runs}\n")

for N in problem_sizes:
    print(f"Processing N={N}...", end='', flush=True)

    tsa_costs = []
    pulp_costs = []
    tsa_times = []
    pulp_times = []
    iterations_list = []
    avg_chain_sizes = []
    max_chain_sizes = []
    min_chain_sizes = []

    successful_cases = 0
    invalid_problems = 0

    for case in range(num_runs):
        problem = generate_problem(N)

        start = time.time()
        tsa_result = TSA(problem)

        if isinstance(tsa_result, tuple):
            tsa_cost, metrics = tsa_result
        else:
            tsa_cost = tsa_result
            metrics = None

        if tsa_cost == -1:
            invalid_problems += 1
            continue

        tsa_times.append(time.time() - start)
        tsa_costs.append(tsa_cost)

        if metrics:
            iterations_list.append(metrics['iterations'])
            avg_chain_sizes.append(metrics['avg_chain_size'])
            max_chain_sizes.append(metrics['max_chain_size'])
            min_chain_sizes.append(metrics['min_chain_size'])

        start = time.time()
        pulp_cost = PuTPS(problem)
        pulp_times.append(time.time() - start)
        pulp_costs.append(pulp_cost)

        successful_cases += 1

    print(f" Done ({successful_cases} successful, {invalid_problems} invalid)")

    tsa_costs = np.array(tsa_costs)
    pulp_costs = np.array(pulp_costs)
    tsa_times = np.array(tsa_times)
    pulp_times = np.array(pulp_times)
    iterations_list = np.array(iterations_list)
    avg_chain_sizes = np.array(avg_chain_sizes)
    max_chain_sizes = np.array(max_chain_sizes)
    min_chain_sizes = np.array(min_chain_sizes)

    # Results for size N
    print(f"\n{'=' * 100}")
    print(f"RESULTS FOR N={N} ({successful_cases} successful, {invalid_problems} invalid)")
    print(f"{'=' * 100}\n")

    # Check for misformed problems
    print(f"{'PROBLEM FEASIBILITY':<30} {'Count':<20} {'Percentage':<20}")
    print("-" * 70)
    print(f"{'Valid problems':<30} {successful_cases:<20} {successful_cases/num_runs*100:<20.2f}%")
    print(f"{'Invalid/infeasible':<30} {invalid_problems:<20} {invalid_problems/num_runs*100:<20.2f}%")

    if successful_cases == 0:
        print("\nNo successful cases to analyze.\n")
        continue

    # Compare costs, should be one and the same
    print(f"\n{'COST ANALYSIS':<30} {'TSA':<20} {'PuLP':<20} {'Difference':<20}")
    print("-" * 90)
    print(
        f"{'Average Cost':<30} {np.mean(tsa_costs):<20.2f} {np.mean(pulp_costs):<20.2f} {np.abs(np.mean(tsa_costs) - np.mean(pulp_costs)):<20.2f}")
    print(
        f"{'Std Dev':<30} {np.std(tsa_costs):<20.2f} {np.std(pulp_costs):<20.2f} {np.abs(np.std(tsa_costs) - np.std(pulp_costs)):<20.2f}")
    print(
        f"{'Min Cost':<30} {np.min(tsa_costs):<20.2f} {np.min(pulp_costs):<20.2f} {np.abs(np.min(tsa_costs) - np.min(pulp_costs)):<20.2f}")
    print(
        f"{'Max Cost':<30} {np.max(tsa_costs):<20.2f} {np.max(pulp_costs):<20.2f} {np.abs(np.max(tsa_costs) - np.max(pulp_costs)):<20.2f}")

    # Algorithm performance metrics
    print(f"\n{'ALGORITHM PERFORMANCE':<30} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 90)
    print(
        f"{'Iterations to optimum':<30} {np.mean(iterations_list):<15.2f} {np.std(iterations_list):<15.2f} {np.min(iterations_list):<15.0f} {np.max(iterations_list):<15.0f}")
    print(
        f"{'Average chain size':<30} {np.mean(avg_chain_sizes):<15.2f} {np.std(avg_chain_sizes):<15.2f} {np.min(avg_chain_sizes):<15.2f} {np.max(avg_chain_sizes):<15.2f}")
    print(
        f"{'Maximum chain size':<30} {np.mean(max_chain_sizes):<15.2f} {np.std(max_chain_sizes):<15.2f} {np.min(max_chain_sizes):<15.0f} {np.max(max_chain_sizes):<15.0f}")
    print(
        f"{'Minimum chain size':<30} {np.mean(min_chain_sizes):<15.2f} {np.std(min_chain_sizes):<15.2f} {np.min(min_chain_sizes):<15.0f} {np.max(min_chain_sizes):<15.0f}")

    # Compare execution time
    tsa_avg_ms = np.mean(tsa_times) * 1000
    pulp_avg_ms = np.mean(pulp_times) * 1000
    speedup = pulp_avg_ms / tsa_avg_ms if tsa_avg_ms > 0 else 0

    print(f"\n{'TIME PERFORMANCE (ms)':<30} {'TSA':<20} {'PuLP':<20} {'Speedup':<20}")
    print("-" * 90)
    print(f"{'Average Time':<30} {tsa_avg_ms:<20.4f} {pulp_avg_ms:<20.4f} {speedup:<20.2f}x")
    print(f"{'Total Time':<30} {np.sum(tsa_times) * 1000:<20.2f} {np.sum(pulp_times) * 1000:<20.2f}")

    # Wins/losses/ties, should be one and the same
    cost_diff = tsa_costs - pulp_costs
    tsa_wins = np.sum(cost_diff < -0.01)
    pulp_wins = np.sum(cost_diff > 0.01)
    ties = np.sum(np.abs(cost_diff) <= 0.01)

    print(f"\n{'SOLUTION QUALITY':<30} {'Count':<20} {'Percentage':<20}")
    print("-" * 70)
    print(f"{'TSA Better':<30} {tsa_wins:<20} {tsa_wins / successful_cases * 100:<20.2f}%")
    print(f"{'PuLP Better':<30} {pulp_wins:<20} {pulp_wins / successful_cases * 100:<20.2f}%")
    print(f"{'Equivalent (±0.01)':<30} {ties:<20} {ties / successful_cases * 100:<20.2f}%")
    print()

print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
