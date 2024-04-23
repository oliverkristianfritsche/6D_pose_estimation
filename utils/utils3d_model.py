import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os
import json
import os
import seaborn as sns
import seaborn as sns

import matplotlib.pyplot as plt

def load_point_clouds(folder_path, json_files, N):
    point_clouds = {'SUV': [], '3x': [], '2x': []}
    titles = {'SUV': [], '3x': [], '2x': []}  # To store titles

    for i, json_file_name in enumerate(json_files[:N]):
        with open(os.path.join(folder_path, json_file_name)) as json_file:
            data = json.load(json_file)
            vertices = np.array(data['vertices'])
            # Adjust vertices for plotting: swap Y and Z, then negate the new Z
            vertices = vertices[:, [0, 2, 1]]  # Swap Y and Z
            vertices[:, 2] = -vertices[:, 2]   # Negate the new Z
            point_clouds[data['car_type']].append(vertices)
            titles[data['car_type']].append(json_file_name)  # Store the file name as title

    return point_clouds, titles

def load_meshes(folder_path, json_files, N):
    meshes = {'SUV': [], '3x': [], '2x': []}
    titles = {'SUV': [], '3x': [], '2x': []}  # To store titles

    for i, json_file_name in enumerate(json_files[:N]):
        with open(os.path.join(folder_path, json_file_name)) as json_file:
            data = json.load(json_file)
            vertices = np.array(data['vertices'])
            faces = np.array(data['faces']) - 1  # Convert to 0-based indexing
            # Adjust vertices for plotting: swap Y and Z, then negate the new Z
            vertices = vertices[:, [0, 2, 1]]  # Swap Y and Z
            vertices[:, 2] = -vertices[:, 2]   # Negate the new Z
            meshes[data['car_type']].append((vertices, faces))
            titles[data['car_type']].append(json_file_name)  # Store the file name as title

    return meshes, titles

def display_point_clouds(point_clouds, titles, num_cols=3, xlim=[-5,5], ylim=[-5,5], zlim=[0,5],colors=None,reference_pc=None):
    sns.set(style="whitegrid")
    num_clouds = len(point_clouds)
    num_rows = (num_clouds + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, pc in enumerate(point_clouds, start=1):
        ax = fig.add_subplot(num_rows, num_cols, i, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.5, c="lightblue" if colors is None else colors[i-1])
        if reference_pc is not None:
            ax.scatter(reference_pc[:, 0], reference_pc[:, 1], reference_pc[:, 2], s=0.5, c="lightyellow")
        ax.set_title(titles[i-1])  # Set the title for each subplot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    
    plt.tight_layout()
    plt.show()

def display_meshes(meshes, titles, num_cols=3,xlim=[-5,5],ylim=[-5,5],zlim=[0,5]):
    sns.set(style="whitegrid")
    num_meshes = len(meshes)
    num_rows = (num_meshes + num_cols - 1) // num_cols
    colors = plt.cm.jet(np.linspace(0, 1, num_meshes))
    fig = plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, (vertices, faces) in enumerate(meshes, start=1):
        ax = fig.add_subplot(num_rows, num_cols, i, projection='3d')
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, linewidth=0.2, antialiased=True, color=colors[i-1])
        ax.set_title(titles[i-1])  # Set the title for each subplot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

    plt.tight_layout()
    plt.show()
