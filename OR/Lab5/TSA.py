import numpy as np
from statistics import mean

def TSA(problem_matrix: np.ndarray[np.float32]):
    iterations = 0
    chain_sizes = []

    # === INIT ===
    # 1. Initial tableau and intervals
    links_matrix = problem_matrix[2:, 2:]
    sources_intervals = problem_matrix[2:, :2]
    sinks_intervals = problem_matrix[:2, 2:].T

    # 3a. Deal with infinities
    low_supply_bounds = sources_intervals[:, 0]
    high_supply_bounds = sources_intervals[:, 1]

    low_demand_bounds = sinks_intervals[:, 0]
    high_demand_bounds = sinks_intervals[:, 1]

    min_total_supply = np.sum(low_supply_bounds)
    max_total_supply = np.sum(high_supply_bounds)

    min_total_demand = np.sum(low_demand_bounds)
    max_total_demand = np.sum(high_demand_bounds)

    if max_total_supply < min_total_demand or min_total_supply > max_total_demand:
        return -1, None

    if (max_total_supply == np.inf) ^ (max_total_demand == np.inf):
        if max_total_supply == np.inf:
            sources_intervals[sources_intervals == np.inf] = max_total_demand
        else:
            sinks_intervals[sinks_intervals == np.inf] = max_total_supply

    elif max_total_supply == np.inf and max_total_demand == np.inf:
        supply_bounds_unknown = np.isinf(high_supply_bounds)
        demand_bounds_unknown = np.isinf(high_demand_bounds)

        max_finite_supply = high_supply_bounds[~supply_bounds_unknown].sum()
        max_finite_demand = high_demand_bounds[~demand_bounds_unknown].sum()

        excess_supply_needed = max(0.0, min_total_demand - max_finite_supply)
        excess_demand_needed = max(0.0, min_total_supply - max_finite_demand)

        high_supply_bounds[supply_bounds_unknown] = low_supply_bounds[supply_bounds_unknown]
        high_demand_bounds[demand_bounds_unknown] = low_demand_bounds[demand_bounds_unknown]

        high_supply_bounds[np.flatnonzero(supply_bounds_unknown)] += excess_supply_needed
        high_demand_bounds[np.flatnonzero(demand_bounds_unknown)] += excess_demand_needed

    # 3b. Deal with big M
    max_link_cost = np.max(links_matrix[links_matrix != np.inf])
    big_M = max_link_cost * 100
    links_matrix[links_matrix == np.inf] = big_M

    # 4. Split
    if np.any(np.concatenate([
        high_supply_bounds < 0,
        high_demand_bounds < 0,
        low_supply_bounds < 0,
        low_demand_bounds < 0,
        high_supply_bounds < low_supply_bounds,
        high_demand_bounds < low_demand_bounds
    ])):
        return -1, None

    high_supply_bounds -= low_supply_bounds
    high_demand_bounds -= low_demand_bounds

    guaranteed_supply = np.flatnonzero(low_supply_bounds)
    guaranteed_demand = np.flatnonzero(low_demand_bounds)

    excess_supply = np.flatnonzero(high_supply_bounds)
    excess_demand = np.flatnonzero(high_demand_bounds)

    split_supply_indices = np.concatenate([guaranteed_supply, excess_supply])
    split_demand_indices = np.concatenate([guaranteed_demand, excess_demand])

    work_array = np.zeros(
        (guaranteed_supply.size + excess_supply.size,
         guaranteed_demand.size + excess_demand.size, 3),
        dtype=np.float32
    )

    supply_mesh, demand_mesh = np.meshgrid(split_supply_indices,
                                           split_demand_indices,
                                           indexing='ij')

    work_array[:, :, 0] = links_matrix[supply_mesh, demand_mesh]

    # 5 Add dummies
    split_guaranteed_supply = low_supply_bounds[guaranteed_supply]
    split_excess_supply = high_supply_bounds[excess_supply]

    split_guaranteed_demand = low_demand_bounds[guaranteed_demand]
    split_excess_demand = high_demand_bounds[excess_demand]

    split_supply = np.concatenate([split_guaranteed_supply, split_excess_supply])
    split_demand = np.concatenate([split_guaranteed_demand, split_excess_demand])

    if split_excess_supply.sum() > 0:
        work_array = np.pad(work_array, ((0, 0), (0, 1), (0, 0)),
                            mode='constant', constant_values=0)

        work_array[guaranteed_supply.size:, -1, 0] = 0
        work_array[:guaranteed_supply.size, -1, 0] = big_M

        split_demand = np.append(split_demand, split_excess_demand.sum())

    if split_excess_demand.sum() > 0:
        work_array = np.pad(work_array, ((0, 1), (0, 0), (0, 0)),
                            mode='constant', constant_values=0)

        work_array[-1, guaranteed_demand.size:, 0] = 0
        work_array[-1, :guaranteed_demand.size, 0] = big_M

        split_supply = np.append(split_supply, split_excess_supply.sum())

    # 6. Balance
    total_split_supply = split_supply.sum()
    total_split_demand = split_demand.sum()

    if total_split_supply > total_split_demand:
        split_demand[-1] += total_split_supply - total_split_demand

    if total_split_demand > total_split_supply:
        split_supply[-1] += total_split_demand - total_split_supply

    problem_size = work_array.shape[0] * work_array.shape[1]

    # 7. Construct workable tableau... DONE!

    # 8. Construct baseline solution with NW corner rule
    supply_left = split_supply.copy()
    demand_left = split_demand.copy()
    for i in range(work_array.shape[0]):
        for j in range(work_array.shape[1]):
            if supply_left[i] == 0 or demand_left[j] == 0:
                continue

            allocation = np.minimum(supply_left[i], demand_left[j])
            work_array[i, j, 1] = allocation
            work_array[i, j, 2] = 1
            supply_left[i] -= allocation
            demand_left[j] -= allocation

    # 9. Compute initial u, v
    u_values = np.full(work_array.shape[0], np.nan, dtype=np.float32)
    v_values = np.full(work_array.shape[1], np.nan, dtype=np.float32)

    u_values[np.argmax(np.sum(work_array[:, :, 2], axis=1))] = 0

    u_computed = np.isfinite(u_values)
    v_computed = np.isfinite(v_values)

    while not (np.all(u_computed) and np.all(v_computed)):
        progress = False
        for i in range(work_array.shape[0]):
            for j in range(work_array.shape[1]):
                if work_array[i, j, 2] == 1:
                    if u_computed[i] and not v_computed[j]:
                        v_values[j] = work_array[i, j, 0] - u_values[i]
                        v_computed[j] = True
                        progress = True
                    elif v_computed[j] and not u_computed[i]:
                        u_values[i] = work_array[i, j, 0] - v_values[j]
                        u_computed[i] = True
                        progress = True
        if not progress:
            break

    if np.logical_or(~np.all(u_computed), ~np.all(v_computed)):
        return -1, None

    # 10. Compute reduced costs
    for i in range(work_array.shape[0]):
        for j in range(work_array.shape[1]):
            if work_array[i, j, 2] == 0:
                work_array[i, j, 1] = work_array[i, j, 0] - (u_values[i] + v_values[j])

    # === LOOP ===
    while np.any(work_array[:, :, 1] < 0):
        iterations += 1

        # 11. Pick node for swap
        nonbasic_mask = work_array[:, :, 2] == 0
        masked_reduced_costs = work_array[:, :, 1].copy()
        masked_reduced_costs[~nonbasic_mask] = np.inf

        target_row_index, target_col_index = np.unravel_index(
            np.argmin(masked_reduced_costs), masked_reduced_costs.shape
        )

        if masked_reduced_costs[target_row_index, target_col_index] >= 0:
            break

        work_array[target_row_index, target_col_index, 1] = 0.0

        # 12. Chain rule (find cycle)
        basic_mask = work_array[:, :, 2] == 1
        num_rows, num_cols = basic_mask.shape

        basic_columns_in_row = [
            np.flatnonzero(basic_mask[row_index]).tolist()
            for row_index in range(num_rows)
        ]
        basic_rows_in_column = [
            np.flatnonzero(basic_mask[:, col_index]).tolist()
            for col_index in range(num_cols)
        ]

        ROW_NODE, COLUMN_NODE = 0, 1

        start_node = (ROW_NODE, target_row_index)
        goal_node = (COLUMN_NODE, target_col_index)

        node_stack = [start_node]
        predecessor = {start_node: None}

        while node_stack:
            current_node_type, current_node_index = node_stack.pop()

            if (current_node_type, current_node_index) == goal_node:
                break

            if current_node_type == ROW_NODE:
                for column_index in basic_columns_in_row[current_node_index]:
                    neighbor_node = (COLUMN_NODE, column_index)
                    if neighbor_node not in predecessor:
                        predecessor[neighbor_node] = (current_node_type, current_node_index)
                        node_stack.append(neighbor_node)
            else:
                for row_index in basic_rows_in_column[current_node_index]:
                    neighbor_node = (ROW_NODE, row_index)
                    if neighbor_node not in predecessor:
                        predecessor[neighbor_node] = (current_node_type, current_node_index)
                        node_stack.append(neighbor_node)

        if goal_node not in predecessor:
            return -1, None

        node_path = []
        current_node = goal_node
        while current_node is not None:
            node_path.append(current_node)
            current_node = predecessor[current_node]
        node_path.reverse()

        cycle = [(target_row_index, target_col_index)]
        for (node_type_a, index_a), (node_type_b, index_b) in zip(node_path[:-1], node_path[1:]):
            if node_type_a == ROW_NODE and node_type_b == COLUMN_NODE:
                row_index, column_index = index_a, index_b
            elif node_type_a == COLUMN_NODE and node_type_b == ROW_NODE:
                row_index, column_index = index_b, index_a
            else:
                return -1, None
            cycle.append((row_index, column_index))

        # Track chain size
        chain_sizes.append(len(cycle))

        minus_cycle_indices = np.arange(1, len(cycle), 2)
        minus_allocations = np.array([
            work_array[cycle[i][0], cycle[i][1], 1] for i in minus_cycle_indices
        ])

        leaving_cycle_index = int(minus_cycle_indices[np.argmin(minus_allocations)])
        leaving_row_index, leaving_col_index = cycle[leaving_cycle_index]

        # 13. Update tableau
        donors = cycle[1::2]
        recipients = cycle[0::2]

        theta = min(work_array[row_index, col_index, 1] for row_index, col_index in donors)

        work_array[leaving_row_index, leaving_col_index, 2] = 0
        work_array[target_row_index, target_col_index, 2] = 1

        for row_index, col_index in recipients:
            work_array[row_index, col_index, 1] += theta

        for row_index, col_index in donors:
            work_array[row_index, col_index, 1] -= theta

        # 14. Recompute u and v, then reduced costs
        u_values[:] = np.nan
        v_values[:] = np.nan
        u_values[np.argmax(np.sum(work_array[:, :, 2], axis=1))] = 0

        u_computed = np.isfinite(u_values)
        v_computed = np.isfinite(v_values)

        while not (np.all(u_computed) and np.all(v_computed)):
            progress = False
            for i in range(work_array.shape[0]):
                for j in range(work_array.shape[1]):
                    if work_array[i, j, 2] == 1:
                        if u_computed[i] and not v_computed[j]:
                            v_values[j] = work_array[i, j, 0] - u_values[i]
                            v_computed[j] = True
                            progress = True
                        elif v_computed[j] and not u_computed[i]:
                            u_values[i] = work_array[i, j, 0] - v_values[j]
                            u_computed[i] = True
                            progress = True
            if not progress:
                break

        if np.logical_or(~np.all(u_computed), ~np.all(v_computed)):
            return -1, None

        for i in range(work_array.shape[0]):
            for j in range(work_array.shape[1]):
                if work_array[i, j, 2] == 0:
                    work_array[i, j, 1] = work_array[i, j, 0] - (u_values[i] + v_values[j])

    # === LOOP END ===
    cost = np.sum(work_array[:, :, 0] * work_array[:, :, 1] * work_array[:, :, 2])

    metrics = {
        'iterations': iterations,
        'chain_sizes': chain_sizes,
        'avg_chain_size': mean(chain_sizes) if chain_sizes else 0,
        'max_chain_size': max(chain_sizes) if chain_sizes else 0,
        'min_chain_size': min(chain_sizes) if chain_sizes else 0,
        'problem_size': problem_size
    }

    if np.any((work_array[..., 2] == 1) & (work_array[..., 0] == big_M)):
        return -1, None

    return cost, metrics
