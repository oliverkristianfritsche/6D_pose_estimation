from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import numpy as np
    

def transform_point_cloud_numpy(pc, transformation_matrix):
    # Add a column of ones to the point cloud to handle translations (homogeneous coordinates)
    homogeneous_pc = np.hstack([pc, np.ones((pc.shape[0], 1))])
    # Apply the transformation matrix to the point cloud
    transformed_homogeneous_pc = homogeneous_pc.dot(transformation_matrix.T)
    # Return only the x, y, z coordinates, not the homogeneous coordinate
    return transformed_homogeneous_pc[:, :3]

def build_transformation_matrix(rotation_vector, translation_vector):
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    return transformation_matrix

def objective_function_rotation(rotation_vector, initial_pc, reference_pc, print_loss=False):
    """
    Objective function that optimizes only rotation.

    Parameters:
        rotation_vector (np.ndarray): The rotation vector to apply.
        initial_pc (np.ndarray): The initial point cloud.
        reference_pc (np.ndarray): The reference point cloud to compare against.
        print_loss (bool, optional): Flag to print the loss after calculation. Defaults to False.

    Returns:
        float: The calculated loss.
    """
    transformation_matrix = build_transformation_matrix(rotation_vector, np.zeros(3))
    transformed_pc = transform_point_cloud_numpy(initial_pc, transformation_matrix)
    loss = chamfer_distance(transformed_pc, reference_pc)

    if print_loss:
        print(f'Loss: {loss}')

    return loss

def chamfer_distance(pc1, pc2):
    dist_matrix = cdist(pc1, pc2, 'euclidean')
    return np.mean(dist_matrix.min(axis=1)) + np.mean(dist_matrix.min(axis=0))