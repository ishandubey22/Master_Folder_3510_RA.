import time
import numpy as np
import cvxpy as cp #this is the library we use for solving the integer linear program
from dataclasses import dataclass
from typing import List, Tuple
@dataclass
class AllocationResult:
    success: bool
    allocation: np.ndarray
    clearing_error: float
    utility: float
class ACEEI:
    def __init__(self, num_students: int, num_courses: int, capacities: np.ndarray, 
                 initial_budgets: np.ndarray, utilities: np.ndarray, 
                 timetable: np.ndarray, step_size: float = 0.1, epsilon: float = 1.0): #can change epsilon here as per requirement(budget perturbation limit)
        self.n = num_students
        self.m = num_courses
        self.capacities = capacities
        self.b0 = initial_budgets
        self.utilities = utilities
        self.timetable = timetable  
        self.delta = step_size
        self.epsilon = epsilon

    def compute_clearing_error(self, allocations: np.ndarray, prices: np.ndarray) -> float:
        # market clearing error
        total_demand = np.sum(allocations, axis=0)
        error = np.zeros(self.m)  
        for j in range(self.m):
            if prices[j] > 0:
                error[j] = abs(total_demand[j] - self.capacities[j])
            else:
                error[j] = max(0, total_demand[j] - self.capacities[j])                
        return np.sum(error)

    def solve_student_allocation(self, student_idx: int, prices: np.ndarray, budget: float) -> AllocationResult:
        x = cp.Variable(self.m, boolean=True)  # binary variable for course selection
        utility = cp.sum(cp.multiply(self.utilities[student_idx], x))
        # constraints
        constraints = [
            cp.sum(cp.multiply(prices, x)) <= budget,  # budget constraint
            x >= 0,
            x <= 1,
        ]
        # preventing timetable clashes
        for t in range(self.timetable.shape[1]):  # iterate over time slots, only one valid schedule is accepted.
            constraints.append(cp.sum(cp.multiply(self.timetable[:, t], x)) <= 1)
        problem = cp.Problem(cp.Maximize(utility), constraints)
        try:
            problem.solve(solver=cp.CBC)
            if problem.status == cp.OPTIMAL:
                return AllocationResult(
                    success=True,
                    allocation=np.round(x.value).astype(int),
                    clearing_error=0,
                    utility=problem.value
                )
        except:
            pass
        return AllocationResult(False, np.zeros(self.m), float('inf'), float('-inf'))

    def find_optimal_budget_perturbation(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        perturbations = cp.Variable(self.n)
        allocations = np.zeros((self.n, self.m))
        # keeping track of the best solution
        best_error = float('inf')
        best_budgets = self.b0.copy()
        best_allocations = allocations.copy()
        # try different budget perturbations within epsilon range, limiting the number of tries to reduce computation.
        for _ in range(10):  
            # generate random perturbations within [-epsilon, epsilon]
            test_perturbations = np.random.uniform(-self.epsilon, self.epsilon, self.n)
            test_budgets = self.b0 + test_perturbations
            # solve allocation for each student with these budgets
            current_allocations = np.zeros((self.n, self.m))
            total_utility = 0
            for i in range(self.n):
                result = self.solve_student_allocation(i, prices, test_budgets[i])
                if result.success:
                    current_allocations[i] = result.allocation
                    total_utility += result.utility
                else:
                    break
            error = self.compute_clearing_error(current_allocations, prices)
            # update best solution if better
            if error < best_error:
                best_error = error
                best_budgets = test_budgets
                best_allocations = current_allocations
        return best_budgets, best_allocations
    def run(self, max_iterations: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: #limited the iterations to 500 for the sake of computation.
        prices = np.ones(self.m)  # initialize prices
        for iteration in range(max_iterations):
            # optimal budget perturbation, calling the ILP we defined earlier.
            budgets, allocations = self.find_optimal_budget_perturbation(prices)
            error = self.compute_clearing_error(allocations, prices) # clearing error
            print(f"Iteration {iteration}: Clearing Error = {error:.4f}") #diagnostic output for seeing how clearing error changes over iterations. insert a # at the beginning to disable it.
            if error < 2:  # we are allowing for 1 seat to be oversubscribed here, can change this as and when required
                return prices, budgets, allocations    
            # tatonment rule (adjusting the prices)
            total_demand = np.sum(allocations, axis=0)
            prices += self.delta * (total_demand - self.capacities)
            prices = np.maximum(prices, 0)  #sanity check, non-negative price. 
        return prices, budgets, allocations
# test case with 10 students and 5 courses
if __name__ == "__main__":
    num_students = 10
    num_courses = 5
    capacities = np.array([2, 2, 2, 2, 1])  
    initial_budgets = np.full(num_students, 100)  # All students start with equal budget - 100.
    #preferences (utility scores) of each course for each student.
    utilities = np.array([
        [8, 2, 5, 1, 4],
        [3, 9, 2, 6, 8],
        [4, 8, 7, 9, 3],
        [2, 6, 4, 8, 5],
        [5, 7, 3, 2, 9],
        [9, 3, 6, 4, 1],
        [2, 4, 8, 7, 6],
        [6, 5, 9, 3, 2],
        [1, 8, 2, 4, 7],
        [7, 9, 1, 5, 3]
    ])
    timetable = np.array([
        [1, 0, 0],  # course 1 happens in time slot 1
        [1, 0, 0],  # course 2 happens in time slot 1
        [0, 1, 0],  # course 3 happens in time slot 2
        [0, 0, 1],  # course 4 happens in time slot 3
        [0, 1, 0],  # course 5 happens in time slot 2
    ])
    solver = ACEEI(
        num_students=num_students,
        num_courses=num_courses,
        capacities=capacities,
        initial_budgets=initial_budgets,
        utilities=utilities,
        timetable=timetable,
        step_size=0.1,
        epsilon=1.0
    )
    # execution time, can be commented out if not needed. for diagnostic purposes.
    start_time = time.time()
    final_prices, final_budgets, final_allocations = solver.run()
    end_time = time.time()
    print("\nFinal Results:")
    print("Prices:", final_prices)
    print("Budgets:", final_budgets)
    print("Allocations:\n", final_allocations)
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
