# -*- coding: utf-8 -*-
import numpy as np
import csv
from docplex.mp.model import Model
import chardet
import time
import cplex

start_time = time.time()    #Record the start time.

class MyCplex:
    def __init__(self):
        self.model = Model(name="myCplex")
        self.vars = []
        self.constraints = []
        self.objective = None
        self.cplex = None

    def define_1d_array(self, size, lower_bound, upper_bounds, var_type, name_prefix):
        vars_array = []
        for i in range(size):
            var_name = f"{name_prefix}[{i}]"
            if var_type == 'integer':
                var = self.model.integer_var(lb=lower_bound, ub=upper_bounds[i], name=var_name)
            else:
                var = self.model.continuous_var(lb=lower_bound, ub=upper_bounds[i], name=var_name)
            vars_array.append(var)
        return vars_array

    def define_3d_array(self, dim1, dim2, dim3):
        array_3d = []
        for i in range(dim1):
            matrix_2d = []
            for j in range(dim2):
                vars_1d = []
                for k in range(dim3):
                    var_name = f"x[{i}][{j}][{k}]"
                    var = self.model.integer_var(lb=0, name=var_name)
                    vars_1d.append(var)
                matrix_2d.append(vars_1d)
            array_3d.append(matrix_2d)
        return array_3d

    def test(self):
        self.objective = self.model.maximize(0)  # 初始化为0

        c1 = self.model.add_range(0, 20, name="constraint_1")
        c2 = self.model.add_range(0, 30, name="constraint_2")
        self.constraints.extend([c1, c2])

        var1 = self.model.continuous_var(lb=0, ub=40, name="var_1")
        var2 = self.model.continuous_var(lb=0, name="var_2")
        var3 = self.model.continuous_var(lb=0, name="var_3")
        self.vars.extend([var1, var2, var3])

        self.objective.set_expr(1 * var1 + 2 * var2 + 3 * var3)

        self.model.add_constraint(var1 + var2 + var3 <= 20)
        self.model.add_constraint(var1 - 3 * var2 + var3 <= 30)

        solution = self.model.solve()

        if solution:
            print(f"Solution status: {solution.solve_status}")
            print(f"Target value: {solution.objective_value}")
            self.model.export_as_lp("model.lp")
        else:
            print("Solution failed.")




def get_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def read_matrix(file_path, dimension):
    """Read matrix data"""
    encoding = get_file_encoding(file_path)
    matrix = np.zeros((dimension, dimension))
    with open(file_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            for j, val in enumerate(row):
                matrix[i][j] = float(val)
    return matrix

def output_matrix_int(matrix, row_size, col_size, output_path):
    """Output the matrix to a file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(row_size):
            writer.writerow(matrix[i][:col_size])


def graph_sfm(file_path, out_path , dimension):
    """The main optimization function"""
    coefficient = -10
    threshold = 0.0000001

    # Read the adjacency matrix
    adjacency_matrix = read_matrix(file_path, dimension)

    # Initialize the CPLEX optimizer.
    cplex_opt = MyCplex()

    # Construct decision variables
    position = dimension // 5
    neighbor_num_set = []
    ortho_neighbor_sets = [[] for _ in range(5)]

    decision_var_index = np.zeros((dimension, dimension), dtype=int)
    var_count = 1

    for i in range(dimension):
        neighbor_num = 0
        ortho_counts = [0] * 5

        for j in range(dimension):
            if adjacency_matrix[i][j] > threshold:
                ortho_idx = j // position
                if ortho_idx < 5:
                    ortho_counts[ortho_idx] += 1
                neighbor_num += 1
                decision_var_index[i][j] = var_count
                var_count += 1

            if adjacency_matrix[j][i] > threshold:
                ortho_idx = j // position
                if ortho_idx < 5:
                    ortho_counts[ortho_idx] += 1
                neighbor_num += 1
                decision_var_index[j][i] = var_count
                var_count += 1

        # Record the number of neighbors for each area.
        ortho_idx = i // position
        if ortho_idx < 5:
            ortho_neighbor_sets[ortho_idx].append(ortho_counts[ortho_idx])

        neighbor_num_set.append(min(neighbor_num, 6))



    # Create decision variables
    decision_vars = cplex_opt.define_1d_array(var_count - 1, 0, [1] * (var_count - 1), 'integer', 'Edge')

    # Build an optimization model
    # 1. Objective function
    # (1) Sum of weights
    total_weight_expr = 0
    for i in range(dimension):
        for j in range(dimension):
            if decision_var_index[i][j] > 0:
                idx = decision_var_index[i][j] - 1
                total_weight_expr += decision_vars[idx] * adjacency_matrix[i][j]


    total_weight_var = cplex_opt.model.continuous_var(lb=0, name="total_weight")
    cplex_opt.model.add_constraint(total_weight_expr - total_weight_var == 0)


    # (2) The total sum of edge quantities
    edge_sum_expr = sum(decision_vars)
    edge_num_var = cplex_opt.model.integer_var(lb=0, ub=var_count - 1, name="edge_num")
    cplex_opt.model.add_constraint(edge_sum_expr - edge_num_var == 0)


    # (3) Comprehensive objective function
    objective_expr = total_weight_var + edge_num_var * coefficient
    cplex_opt.model.maximize(objective_expr)


    # 2. Constraint conditions
    # (1) Minimum neighbor constraint
    for i in range(dimension):
        neighbor_constraint = 0
        for j in range(dimension):
            idx = -1
            if adjacency_matrix[i][j] > threshold:
                idx = decision_var_index[i][j] - 1
            if adjacency_matrix[j][i] > threshold:
                idx = decision_var_index[j][i] - 1

            if idx >= 0:
                neighbor_constraint += decision_vars[idx]


        # Add constraint: Each point must have at least t neighbors.
        cplex_opt.model.add_constraint(neighbor_constraint >= neighbor_num_set[i])


    # (2) Same-viewpoint matching constraint
    for k in range(5):
        offset = k * position
        for i in range(offset, offset + position):
            ortho_neighbor = ortho_neighbor_sets[k][i - offset]
            ortho_neighbor = min(ortho_neighbor, 1)

            ortho_constraint = 0
            for j in range(offset, offset + position):
                idx = -1
                if adjacency_matrix[i][j] > threshold:
                    idx = decision_var_index[i][j] - 1
                if adjacency_matrix[j][i] > threshold:
                    idx = decision_var_index[j][i] - 1

                if idx >= 0:
                    ortho_constraint += decision_vars[idx]

            # Add constraint: Each point must have at least one match in the same category of perspectives.
            cplex_opt.model.add_constraint(ortho_constraint >= ortho_neighbor)


    # Solving model
    solution = cplex_opt.model.solve()

    # Output result
    if solution:
        print(f"Solution status: {solution.solve_status}")
        print(f"Target value: {solution.objective_value}")
        cplex_opt.model.export_as_lp("model.lp")


        # Obtain the optimization results
        optimized_result = [1 if solution.get_value(var) > 0.1 else 0 for var in decision_vars]


        # Construct the optimized matrix.
        optimized_matrix = np.zeros((dimension, dimension), dtype=int)
        left_edges = 0


        for i in range(dimension):
            for j in range(dimension):
                if adjacency_matrix[i][j] > threshold:
                    idx = decision_var_index[i][j] - 1
                    optimized_matrix[i][j] = optimized_result[idx]
                    if optimized_matrix[i][j] > 0:
                        left_edges += 1


        # Output the matrix to a file.
        output_path = out_path.replace(".csv", ".csv")
        output_matrix_int(optimized_matrix, dimension, dimension, output_path)

        print(f"The remaining side: {left_edges} / {var_count - 1}")
    else:
        print("Solution failed.")


if __name__ == "__main__":
    file_path = "../input/wuhan_test.csv"
    out_path = "../result/SCN.csv"
    graph_sfm(file_path, out_path, 30)

end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Computation time consumption
print(f"Operation time consumption：{elapsed_time} s")