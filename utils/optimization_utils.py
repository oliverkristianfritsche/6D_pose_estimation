from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
    

def transform_point_cloud_numpy(pc, transformation_matrix):
    # Add a column of ones to the point cloud to handle translations (homogeneous coordinates)
    homogeneous_pc = np.hstack([pc, np.ones((pc.shape[0], 1))])
    # Apply the transformation matrix to the point cloud
    transformed_homogeneous_pc = homogeneous_pc.dot(transformation_matrix.T)
    # Return only the x, y, z coordinates, not the homogeneous coordinate
    return transformed_homogeneous_pc[:, :3]

def build_transformation_matrix(rotation_vector, translation_vector):
    # Create a 3x3 rotation matrix from the rotation vector
    theta = np.linalg.norm(rotation_vector)
    if theta > 0:
        normalized_axis = rotation_vector / theta
        K = np.array([
            [0, -normalized_axis[2], normalized_axis[1]],
            [normalized_axis[2], 0, -normalized_axis[0]],
            [-normalized_axis[1], normalized_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    else:
        R = np.eye(3)

    # Create a 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation_vector

    return T

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
    transformed_pc = transform_point_cloud_numpy(reference_pc, transformation_matrix)
    loss = chamfer_distance(transformed_pc, initial_pc)

    if print_loss:
        print(f'Loss: {loss}')

    return loss

def objective_function(params, initial_pc, reference_pc):
    """
    Calculate the Chamfer distance between two point clouds after transforming
    one of them with the provided rotation and translation parameters.

    Parameters:
        params (np.ndarray): Array of six parameters where the first three are 
                             rotation vector and the next three are translation vector.
        initial_pc (np.ndarray): The point cloud to be transformed.
        reference_pc (np.ndarray): The reference point cloud.

    Returns:
        float: The Chamfer distance as the loss value.
    """
    # Split parameters into rotation vector and translation vector
    rotation_vector = params[:3]
    translation_vector = params[3:]
    # Build transformation matrix from parameters
    transformation_matrix = build_transformation_matrix(rotation_vector, translation_vector)
    # Transform the initial point cloud
    transformed_pc = transform_point_cloud_numpy(initial_pc, transformation_matrix)
    # Calculate the Chamfer distance
    loss = chamfer_distance(transformed_pc, reference_pc)
    return loss



def chamfer_distance(pc1, pc2):
    dist_matrix = cdist(pc1, pc2, 'euclidean')
    return np.mean(dist_matrix.min(axis=1)) + np.mean(dist_matrix.min(axis=0))

# Function to calculate volume of point cloud using convex hull
def calculate_volume(point_cloud):
    hull = ConvexHull(point_cloud)
    return hull.volume

def calculate_angular_error(rot1, rot2):
    norm_rot1, norm_rot2 = np.linalg.norm(rot1), np.linalg.norm(rot2)
    if norm_rot1 == 0 or norm_rot2 == 0:
        return 0
    cos_theta = np.clip(np.dot(rot1/norm_rot1, rot2/norm_rot2), -1.0, 1.0)
    return np.arccos(cos_theta) * (180 / np.pi)

def calculate_individual_angular_errors(rot1, rot2):
    # Convert rotations from radians to degrees
    rot1_degrees = np.degrees(rot1)
    rot2_degrees = np.degrees(rot2)

    # Calculate absolute error modulo 360 degrees
    yaw_error = np.abs(rot1_degrees[0] - rot2_degrees[0]) % 360
    pitch_error = np.abs(rot1_degrees[1] - rot2_degrees[1]) % 360
    roll_error = np.abs(rot1_degrees[2] - rot2_degrees[2]) % 360

    # Adjust for minimum angular difference
    yaw_error = min(yaw_error, 360 - yaw_error)
    pitch_error = min(pitch_error, 360 - pitch_error)
    roll_error = min(roll_error, 360 - roll_error)

    # Return the errors as a list of floats
    return [yaw_error, pitch_error, roll_error]

def optimize_alignment(initial_pc, reference_pc, actual_rotation, num_trials=10):
    best_loss = float('inf')
    best_transformation = None
    list_of_dicts = []

    for trial_number in range(num_trials):
        trial_loss_history, trial_angular_errors = [], []

        def callback(xk):
            loss = objective_function_rotation(xk, initial_pc, reference_pc)
            angular_error = calculate_individual_angular_errors(xk, actual_rotation)
            trial_loss_history.append(loss)
            trial_angular_errors.append(angular_error)

        initial_rotation = np.random.rand(3) * 2 * np.pi
        res = minimize(objective_function_rotation, initial_rotation, args=(initial_pc, reference_pc), callback=callback)

        if res.fun < best_loss:
            best_loss = res.fun
            best_transformation = res.x

        #add ti data so its columns are [run,trial,best_loss,best_transformation,loss_histories(iterations,loss at iteration),angular_error_histories(iterations,[yaw,pitch,roll] at iteration)]
        data = {
            'Run': -1, # Placeholder for run number
            'Trial': trial_number + 1,
            'Trial Best Transformation': best_transformation,
            'Trial Initial Rotation': initial_rotation,
            'Loss History': trial_loss_history,
            'Angular Error History': trial_angular_errors
        }

        list_of_dicts.append(data)

    return list_of_dicts

def multiple_optimizations(initial_pc, reference_pc, actual_rotation, num_runs=50, num_trials=10):
    results_df = pd.DataFrame()

    for run in range(num_runs):
        data = optimize_alignment(initial_pc, reference_pc, actual_rotation, num_trials)
        # Add run number to each dictionary
        for d in data:
            d['Run'] = run + 1
        
        results_df = pd.concat([results_df, pd.DataFrame(data)], ignore_index=True)

    return results_df
