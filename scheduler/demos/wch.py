import numpy as np
import random

objectives = [
    lambda plan: sum(1 for _ in plan),  # Maximize service scale
    lambda plan: sum(service['demands']['cpu'] for _, service, _ in plan),  # Example accuracy proxy
    lambda plan: sum(resource_constraints[edge]['cpu'] for _, _, edge in plan),  # Remaining resources proxy
]

def generate_weight_vectors(num_weights, num_objectives):
    """
    Generate uniformly distributed weight vectors for scalarization.
    """
    weights = np.random.dirichlet(np.ones(num_objectives), size=num_weights)
    return weights

def feasible_deployments(devices, services, edge_nodes, resource_constraints):
    """
    Generate feasible service deployments based on resource constraints.
    """
    feasible = []
    for device in devices:
        for service in services:
            for edge_node in edge_nodes:
                if satisfies_constraints(service, edge_node, resource_constraints):
                    feasible.append((device, service, edge_node))
    return feasible

def satisfies_constraints(service, edge_node, resource_constraints):
    """
    Check if a service deployment satisfies resource constraints.
    """
    for resource, demand in service['demands'].items():
        if demand > resource_constraints[edge_node][resource]:
            return False
    return True

def scalarized_objective(deployment_plan, weights, objectives):
    """
    Compute the scalarized objective value based on weights.
    """
    values = [objective(deployment_plan) for objective in objectives]
    return max(weights[i] * (1 - value) for i, value in enumerate(values))

def wch(devices, services, edge_nodes, resource_constraints, num_weights=1000):
    """
    Weighted Constructive Heuristic algorithm for multi-objective optimization.
    """
    # Step 1: Generate weight vectors
    weights = generate_weight_vectors(num_weights, len(objectives))

    # Step 2: Initialize solutions
    solution_set = []

    for weight in weights:
        current_plan = []
        remaining_constraints = resource_constraints.copy()

        for device in devices:
            feasible = feasible_deployments([device], services, edge_nodes, remaining_constraints)
            if not feasible:
                break

            # Step 3: Select the best deployment based on scalarized objective
            best_deployment = None
            best_value = float('inf')

            for deployment in feasible:
                temp_plan = current_plan + [deployment]
                value = scalarized_objective(temp_plan, weight, objectives)
                if value < best_value:
                    best_value = value
                    best_deployment = deployment

            if best_deployment:
                current_plan.append(best_deployment)
                update_constraints(best_deployment, remaining_constraints)

        solution_set.append(current_plan)

    # Step 4: Filter non-dominated solutions
    return filter_non_dominated_solutions(solution_set)

def update_constraints(deployment, constraints):
    """
    Update resource constraints based on the deployment.
    """
    device, service, edge_node = deployment
    for resource, demand in service['demands'].items():
        constraints[edge_node][resource] -= demand

def filter_non_dominated_solutions(solutions):
    """
    Filter the set of solutions to only include non-dominated solutions.
    """
    non_dominated = []
    for solution in solutions:
        if all(not dominates(other, solution) for other in solutions if other != solution):
            non_dominated.append(solution)
    return non_dominated

def dominates(solution_a, solution_b):
    """
    Check if solution_a dominates solution_b.
    """
    return all(a >= b for a, b in zip(solution_a, solution_b)) and any(a > b for a, b in zip(solution_a, solution_b))

# Example usage
if __name__ == "__main__":
    devices = ["device1", "device2", "device3"]
    services = [
        {"name": "service1", "demands": {"cpu": 2, "ram": 4}},
        {"name": "service2", "demands": {"cpu": 1, "ram": 2}},
    ]
    edge_nodes = ["node1", "node2"]
    resource_constraints = {
        "node1": {"cpu": 4, "ram": 8},
        "node2": {"cpu": 3, "ram": 6},
    }

    solutions = wch(devices, services, edge_nodes, resource_constraints)
    print("Non-dominated solutions:", solutions)
