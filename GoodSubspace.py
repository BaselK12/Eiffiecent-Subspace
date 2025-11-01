import numpy as np
from scipy.linalg import null_space
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# Hatem Tayim, 211731450
# Basel Karkabee, 322629973

# the function get a vector, and returns its Euclidean norm
# input: p (np.array)
# output: float
def euclidean_norm(p):
    return np.linalg.norm(p)


# the function gets 2 vector a,b, and it returns their dot product.
# the return type is a scalar
# Input: a (np.array), b (array)
# Output: float
def dot_product(a, b):
    return np.dot(a, b)


# the function gets two vectors p, v, and returns the component of p orthogonal to v
# it uses the equation: p - (p · v) * v
# Input:  p (np.array), v (np.array)
# Output: (np.array)
def project_onto_subspace(p, v): #
    return p - dot_product(p, v) * v


# this function gets point which are represented as a list of vectors
# it picks a random vector (higher probability for longer vectors)
# returns the normalized vector
# Input:  points List[np.array]
# Output: (np.array)
def sample_point_direction(points):
    # computes the length of each point (vector)
    dists = np.array([np.linalg.norm(p) for p in points])
    if np.sum(dists) == 0:
        # return a vector with same values
        return np.ones(points[0].shape) / np.sqrt(points[0].shape[0])
    # computes the probability of each vector based on length
    probs = dists / np.sum(dists)
    idx = np.random.choice(len(points), p=probs) # pick random vector
    return points[idx] / np.linalg.norm(points[idx]) # return the normalized vector


# this function gets point which are represented as a list of vectors, and also an error parameter (epsilon)
# the purpose of this function, is to mix new random direction (vectors) with previous ones
# it returns one random vector from the pool
# Input:  points List[np.array], epsilon (float)
# Output: (np.array)
def sample_line_sequence(points, epsilon):

    # computes the number of lines, based on the formula from the paper (using epsilon)
    num_lines = int(np.ceil((1 / epsilon) * np.log(1 / epsilon))) + 1
    lines = []

    # uses the previous function to get a vector randomly
    v = sample_point_direction(points)
    lines.append(v)

    for _ in range(1, num_lines):
        u = lines[-1]  # last direction in the list
        v = sample_point_direction(points) # we generate new direction

        # with 50% chance flip the direction
        if np.random.rand() < 0.5:
            u = -u

        # we pick a random spot between u and v
        alpha = np.random.rand()
        new_dir = alpha * u + (1 - alpha) * v

        # we normalize the new direction
        new_dir = new_dir / np.linalg.norm(new_dir)
        lines.append(new_dir)

    # return one line at random from pool of vectors
    return lines[np.random.randint(0, len(lines))]


# this function implements the fit measure as stated in the paper
# this function gets a subspace, and a pool of points (list of vectors), and a parameter tau
# it computes the RD(F,P) and returns it
# Input:  subspace (List[np.ndarray]), points List[np.array], tau (float)
# Output: (float)
def rd_tau(subspace, points, tau=2):
    # we check if the subspace is a single vector
    if isinstance(subspace, np.ndarray) and subspace.ndim == 1:
        # calculate the distance from the points to the vector
        dists = [np.linalg.norm(p - dot_product(p, subspace) * subspace) for p in points]
    else:
        basis = np.array(subspace).T
        dists = []
        for p in points:
            # project p onto the subspace
            proj = basis @ (basis.T @ p)
            # the distance from p to the subspace
            dist = np.linalg.norm(p - proj)
            dists.append(dist)
    dists = np.array(dists)

    # we return the value of error based on tau as in the paper
    if tau == np.inf:
        # max distances
        return np.max(dists)
    elif tau == 1:
        # sum of distances
        return np.sum(dists)
    else:
        # L_tau norm
        return (np.sum(dists ** tau)) ** (1.0 / tau)


# this function gets a vector and a basis
# it makes the vector orthogonal to the basis and  then normalize the vector
# it returns a new basis which is the basis + the new vector
# Input:  vector (np.array), basis (List[np.ndarray])
# Output: List[np.array]
def span(vector, basis):
    if basis is None:
        # if there is no basis, then just return the vector itself
        return [vector]
    for g in basis:
        # make the vector orthogonal to the basis
        vector = vector - dot_product(vector, g) * g
    # normalize the vector
    vector = vector / euclidean_norm(vector)

    # return the new basis = basis + the new vector
    return [vector] + list(basis)


# this function gets a pool of points (list of vectors), and a parameter tol
# it checks if all the points are identical up to the tol
# and returns true if all are the same, otherwise it will return false
# Input:  points (List[np.array]), tol (float)
# Output: (bool)
def all_points_identical(points, tol=1e-10):
    return np.allclose(points, points[0], atol=tol)


# this function gets a vector v (with dimension d)
# it returns a set of (d-1) vectors that are orthonormal, and span the space orthogonal to v
# we use null_space cause it finds all x such that vx=0
# Input:  v (np.array)
# Output: (np.array)
def null_space_basis(v):
    return null_space(v.reshape(1, -1))  # shape: (d, d-1)


# this function gets basis, points (list of vectors), the target k, the error (epsilon) and tau
# this function finds a good k-dimensional subspace that fits the points
# Input:  basis (np.array), points (List[np.array]),
#         k (int), epsilon (float), tau (float), points_orig (List[np.array])
# Output: (List[np.array])
def good_subspace(basis, points, k, epsilon, tau=2, points_orig=None):

    # If no subspace is given, we start with full space
    if basis is None:
        d = points[0].shape[0]
        basis = np.eye(d).T  # (d, d) columns are orthonormal basis vectors
        points_orig = [basis.T @ p for p in points]
    else:
        points_orig = points  # Only pass projected points through recursion

    # If all projected points are identical, return any k-dim subspace of S
    if all_points_identical(points_orig):
        return [basis[:, i] for i in range(k)]

    # Base case (k=1): Find best 1D subspace in S
    if k == 1:
        best_v = None
        best_error = float('inf')
        num_samples = int(np.ceil((1 / epsilon) * np.log(1 / epsilon))) + 1
        for _ in range(num_samples):
            v = sample_line_sequence(points_orig, epsilon)
            err = rd_tau(v, points_orig, tau)
            if err < best_error:
                best_error = err
                best_v = v
        return [basis @ best_v]

    # Paper's recursive step: generate all candidates, pick one at random
    num_samples = int(np.ceil((1 / epsilon) * np.log(1 / epsilon))) + 1

    # generate candidate direction (vectors)
    candidates = [sample_line_sequence(points_orig, epsilon) for _ in range(num_samples)]
    v = candidates[np.random.randint(len(candidates))]

    # project all points to be orthogonal to v
    points_orig_minus_v = [project_onto_subspace(p, v) for p in points_orig]
    # compute an orthonormal basis for the subspace orthogonal to v
    q = null_space_basis(v)  # (m, m-1)
    # project the points and the basis into the new subspace
    points_next = [q.T @ p for p in points_orig_minus_v]  # (m-1,)
    basis_next = basis @ q  # (d, m-1)

    # recursive call to find the remaining basis vectors
    sub_basis = good_subspace(basis_next, points_next, k - 1, epsilon, tau, points_orig)

    # returns the new direction + the basis
    return [basis @ v] + sub_basis


# this function gets a basis and data (which are the points)
# it projects the data onto the subspace spanned by the basis
# it computes the projection error and returns it
# Input:  basis (List[np.array]), data (np.ndarray)
# Output: (float)
def projection_error(basis, data):
    basis_matrix = np.column_stack(basis)
    projections = basis_matrix @ (basis_matrix.T @ data.T)
    return np.linalg.norm(data - projections.T, 'fro')


# this function gets a basis
# it checks if the basis is orthonormal or not
# if it is orthonormal it returns true, otherwise it returns false
# Input:  basis (List[np.array])
# Output: (bool)
def is_orthonormal(basis):
    basis = np.column_stack(basis)
    return np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-6)


####################################################################################################


# in this function we run both our subspace algorithm and PCA on the same dataset
# we collect data to compare them, and return the data
# Input:
#   points (List[np.array]), x_centered (np.array), k (int)
#   epsilon (float), tau (float)
# Output:
#   dict with error and runtime metrics for both algorithms
def run_algorithms(points, x_centered, k, epsilon, tau):

    #  my algorithm
    t1 = time.process_time()
    t0 = time.time()

    # we run the good subspace algorithm with the parameters
    basis_alg = good_subspace(None, points, k, epsilon=epsilon, tau=tau)

    elapsed_alg = time.time() - t0
    cpu_time = time.process_time() - t1

    err_alg = rd_tau(basis_alg, points, tau=tau) # computes the error
    proj_err_alg = projection_error(basis_alg, x_centered) # computes the projection error

    # PCA
    pca = PCA(n_components=k)
    pca.fit(x_centered)
    basis_pca = pca.components_.T
    # we make it as a list to match our functions
    basis_pca_list = [basis_pca[:, i] for i in range(basis_pca.shape[1])]

    err_pca = rd_tau(basis_pca_list, points, tau=tau) # computes the error
    proj_err_pca = projection_error(basis_pca_list, x_centered) # computes the projection error

    return {
        "err_alg": err_alg,
        "err_pca": err_pca,
        "elapsed_alg": elapsed_alg,
        "cpu_time_alg": cpu_time,
        "proj_err_alg": proj_err_alg,
        "proj_err_pca": proj_err_pca
    }


# this function runs the algorithm on many k values, from 1 to d
# and collects the date and then plot a graph to compare the result
# of our algorithm vs PCA
def experiment_k():

    epsilon = 0.03
    tau = 2
    d = 10
    m = 15

    # init the points
    # wine data is a machine learning data that are saved in python
    # contains 178 sample (we use as points)
    # and 13 features (we use as dimension)
    x_full = load_wine().data

    # we take as many dimensions as we want up to 13
    col_idx = np.random.choice(x_full.shape[1], d, replace=False)
    x = x_full[:, col_idx]
    x_centered = x - np.mean(x, axis=0)

    # we take as many points as we want up to 178
    row_idx = np.random.choice(x_centered.shape[0], m, replace=False)
    points = [x_centered[i] for i in row_idx]
    d = x_centered.shape[1]

    print("parameters:")
    print(f"number of points = {m},  dimension = {d}, epsilon = {epsilon}, tau = {tau}")

    k_values = range(1, min(17, d+1))

    errors_alg = []
    errors_pca = []
    times_alg = []
    cpu_time_alg = []
    proj_errors_alg = []
    proj_errors_pca = []

    for k in k_values:

        data = run_algorithms(points, x_centered, k, epsilon, tau)
        errors_alg.append(data["err_alg"])
        errors_pca.append(data["err_pca"])
        times_alg.append(data["elapsed_alg"])
        cpu_time_alg.append(data["cpu_time_alg"])
        proj_errors_alg.append(data["proj_err_alg"])
        proj_errors_pca.append(data["proj_err_pca"])

    results = {
        "k_values": k_values,
        "errors_alg": errors_alg,
        "errors_pca": errors_pca,
        "times_alg": times_alg,
        "cpu_alg" : cpu_time_alg,
        "proj_errors_alg": proj_errors_alg,
        "proj_errors_pca": proj_errors_pca,
    }

    # these functions plot the data we collected
    plot_error(results, "k")
    plot_runtime(results, "k")
    plot_projection_error(results, "k")



# this function runs the algorithm on many epsilon values, from [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
# and collects the date and then plot a graph to compare the result
# of our algorithm vs PCA
def experiment_epsilon():

    k = 5
    tau = 2
    d = 10
    m = 15

    # init the points
    # wine data is a machine learning data that are saved in python
    # contains 178 sample (we use as points)
    # and 13 features (we use as dimension)
    x_full = load_wine().data

    # we take as many dimensions as we want up to 13
    col_idx = np.random.choice(x_full.shape[1], d, replace=False)
    x = x_full[:, col_idx]
    x_centered = x - np.mean(x, axis=0)

    # we take as many points as we want up to 178
    row_idx = np.random.choice(x_centered.shape[0], m, replace=False)
    points = [x_centered[i] for i in row_idx]

    print("parameters:")
    print(f"number of points = {m},  dimension = {d}, k = {k}, tau = {tau}")
    epsilon_values = [0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

    errors_alg = []
    errors_pca = []
    times_alg = []
    cpu_time_alg = []
    proj_errors_alg = []
    proj_errors_pca = []
    for epsilon in epsilon_values:

        data = run_algorithms(points, x_centered, k, epsilon, tau)
        errors_alg.append(data["err_alg"])
        errors_pca.append(data["err_pca"])
        times_alg.append(data["elapsed_alg"])
        cpu_time_alg.append(data["cpu_time_alg"])
        proj_errors_alg.append(data["proj_err_alg"])
        proj_errors_pca.append(data["proj_err_pca"])

    results = {
        "epsilon_values": epsilon_values,
        "errors_alg": errors_alg,
        "errors_pca": errors_pca,
        "times_alg": times_alg,
        "cpu_alg" : cpu_time_alg,
        "proj_errors_alg": proj_errors_alg,
        "proj_errors_pca": proj_errors_pca,
    }

    # these functions plot the data we collected
    plot_error(results, "epsilon")
    plot_runtime(results, "epsilon")
    plot_projection_error(results, "epsilon")


# this function runs the algorithm on different number of points
# from [5, 8, 10, 12, 15, 20, 25, 30, 50]
# and collects the date and then plot a graph to compare the result
# of our algorithm vs PCA
def experiment_points():

    k = 5
    epsilon = 0.03
    tau = 2
    d = 10

    # init the points
    # wine data is a machine learning data that are saved in python
    # contains 178 sample (we use as points)
    # and 13 features (we use as dimension)
    x_full = load_wine().data

    # we take as many dimensions as we want up to 13
    col_idx = np.random.choice(x_full.shape[1], d, replace=False)
    x = x_full[:, col_idx]
    x_centered = x - np.mean(x, axis=0)

    print("parameters:")
    print(f"dimensions = {d}, k = {k}, epsilon = {epsilon}, tau = {tau}")

    errors_alg = []
    errors_pca = []
    times_alg = []
    cpu_time_alg = []
    proj_errors_alg = []
    proj_errors_pca = []

    points_values = [5, 8, 10, 12, 15, 20, 25, 30, 50]

    for m in points_values:

        # each time in the loop we choose a different number of points
        row_idx = np.random.choice(x_centered.shape[0], m, replace=False)
        points = [x_centered[i] for i in row_idx]

        data = run_algorithms(points, x_centered, k, epsilon, tau)
        errors_alg.append(data["err_alg"])
        errors_pca.append(data["err_pca"])
        times_alg.append(data["elapsed_alg"])
        cpu_time_alg.append(data["cpu_time_alg"])
        proj_errors_alg.append(data["proj_err_alg"])
        proj_errors_pca.append(data["proj_err_pca"])

    results = {
        "points_values": points_values,
        "errors_alg": errors_alg,
        "errors_pca": errors_pca,
        "times_alg": times_alg,
        "cpu_alg" : cpu_time_alg,
        "proj_errors_alg": proj_errors_alg,
        "proj_errors_pca": proj_errors_pca,
    }

    # these functions plot the data we collected
    plot_error(results, "points")
    plot_runtime(results, "points")
    plot_projection_error(results, "points")


# this function runs the algorithm on different values of dimensions
# from 5 to 13
# and collects the date and then plot a graph to compare the result
# of our algorithm vs PCA
def experiment_dimension():

    # init the points
    # wine data is a machine learning data that are saved in python
    # contains 178 sample (we use as points)
    # and 13 features (we use as dimension)
    x_full = load_wine().data
    m = 15
    k = 5
    epsilon = 0.03
    tau = 2

    print("parameters:")
    print(f"number of points = {m}, k = {k}, epsilon = {epsilon}, tau = {tau}")

    errors_alg = []
    errors_pca = []
    times_alg = []
    cpu_time_alg = []
    proj_errors_alg = []
    proj_errors_pca = []

    dimension_values = range(k, 13)

    for d in dimension_values:

        # at each loop we choose
        col_idx = np.random.choice(x_full.shape[1], d, replace=False)
        x = x_full[:, col_idx]
        x_centered = x - np.mean(x, axis=0)
        row_idx = np.random.choice(x_centered.shape[0], m, replace=False)
        points = [x_centered[i] for i in row_idx]

        data = run_algorithms(points, x_centered, k, epsilon, tau)
        errors_alg.append(data["err_alg"])
        errors_pca.append(data["err_pca"])
        times_alg.append(data["elapsed_alg"])
        cpu_time_alg.append(data["cpu_time_alg"])
        proj_errors_alg.append(data["proj_err_alg"])
        proj_errors_pca.append(data["proj_err_pca"])

    results = {
        "dimension_values": list(dimension_values),
        "errors_alg": errors_alg,
        "errors_pca": errors_pca,
        "times_alg": times_alg,
        "cpu_alg" : cpu_time_alg,
        "proj_errors_alg": proj_errors_alg,
        "proj_errors_pca": proj_errors_pca,
    }

    plot_error(results, "dimension")
    plot_runtime(results, "dimension")
    plot_projection_error(results, "dimension")


# Plot error curves for both algorithms as a function of the given variable.
# Input:
#   results: dict (contains results arrays to plot)
#   variable: str (the name of the variable to plot)
def plot_error(results,variable):
    plt.figure(figsize=(8,5))
    x = results[variable + "_values"]
    plt.plot(x, results["errors_alg"], label="My Algorithm", marker="o")
    plt.plot(x, results["errors_pca"], label="PCA", marker="x")
    plt.xlabel(f"{variable}")
    plt.ylabel(f"Error")
    plt.title(f"Error vs {variable}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot runtime curves for both algorithms as a function of the given variable.
# Input:
#   results: dict (contains results arrays to plot)
#   variable: str (the name of the variable to plot)
def plot_runtime(results, variable):
    plt.figure(figsize=(8,5))
    x = results[variable + "_values"]
    plt.plot(x, results["times_alg"], label="runtime", marker="o")
    plt.plot(x, results["cpu_alg"], label="cpu runtime", marker="o")
    plt.xlabel(f"{variable}")
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime vs {variable}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot projection error curves for both algorithms as a function of the given variable.
# Input:
#   results: dict (contains results arrays to plot)
#   variable: str (the name of the variable to plot)
def plot_projection_error(results, variable):
    plt.figure(figsize=(8,5))
    x = results[variable + "_values"]
    plt.plot(x, results["proj_errors_alg"], label="My Algorithm", marker="o")
    plt.plot(x, results["proj_errors_pca"], label="PCA", marker="x")
    plt.xlabel(f"{variable}")
    plt.ylabel("Projection Error")
    plt.title(f"Projection Error vs {variable}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# this functions runs my algorithm based on data from the user
# also it prints the error data
def single_run():

    # Prompt user for parameters
    print("\nEnter parameters for a single run:")
    d = int(input("Number of dimensions (d) up to 13: "))
    m = int(input("Number of points (m) up to 178: "))
    k = int(input("Target subspace dimension (k): "))
    epsilon = float(input("Approximation parameter (epsilon): "))
    tau_input = input("Error parameter (tau, can be 1, 2, or 'inf'): ").strip()
    tau = float('inf') if tau_input == 'inf' else float(tau_input)

    # Initialize the points
    x_full = load_wine().data
    # Select d dimensions randomly
    col_idx = np.random.choice(x_full.shape[1], d, replace=False)
    x = x_full[:, col_idx]
    x_centered = x - np.mean(x, axis=0)
    # Select m points randomly
    row_idx = np.random.choice(x_centered.shape[0], m, replace=False)
    points = [x_centered[i] for i in row_idx]

    # Run the algorithms
    data = run_algorithms(points, x_centered, k, epsilon, tau)

    # Print results
    print("\nResults for Single Run:")
    print(f"Algorithm error (RD_tau): {data['err_alg']:.4f}")
    print(f"PCA error (RD_tau): {data['err_pca']:.4f}")
    print(f"Algorithm projection error: {data['proj_err_alg']:.4f}")
    print(f"PCA projection error: {data['proj_err_pca']:.4f}")
    print(f"Algorithm runtime: {data['elapsed_alg']:.4f} sec")
    print(f"Algorithm CPU time: {data['cpu_time_alg']:.4f} sec")



def main():

    while True:
        print("Which experiment do you want to run?")
        print("1: experiment on k values")
        print("2: experiment on epsilon values")
        print("3: experiment on number of points")
        print("4: experiment on number of dimension")
        print("5: a single run of the algorithm")
        choice = input("Enter the number of your choice (1-5): ").strip()

        if choice == "1":
            experiment_k()
            break
        elif choice == "2":
            experiment_epsilon()
            break
        elif choice == "3":
            experiment_points()
            break
        elif choice == "4":
            experiment_dimension()
            break
        elif choice == "5":
            single_run()
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.\n")




if __name__ == "__main__":
    main()

