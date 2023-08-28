from scipy.optimize import linear_sum_assignment
import numpy as np

def find_matching(A, B):
    # Compute the distance matrix between points in A and B
    dist_matrix = np.sqrt(np.sum((A[:, np.newaxis, :] - B)**2, axis=2))
    print(dist_matrix)
    # Use the Hungarian algorithm to find the optimal assignment of points in A to points in B
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    
    # Return the indices of the assigned points in A and B
    return row_ind, col_ind

# Example usage
A = np.random.rand(4, 3)  # Point cloud A with 10 points in 3D space
B = np.random.rand(4, 3)  # Point cloud B with 15 points in 3D space
row_ind, col_ind = find_matching(A, B)
print(row_ind, col_ind)
print(A)
print(B)
print(np.linalg.norm(A[0] - B[1]))