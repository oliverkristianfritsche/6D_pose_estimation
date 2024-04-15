import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R



def load_point_clouds(folder_path, json_files, N):

    import json

    import os

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

    import json

    import os

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



def display_point_clouds(point_clouds, titles, num_cols=3):

    import seaborn as sns

    sns.set(style="whitegrid")

    num_clouds = len(point_clouds)

    num_rows = (num_clouds + num_cols - 1) // num_cols

    colors = plt.cm.jet(np.linspace(0, 1, num_clouds))

    fig = plt.figure(figsize=(num_cols * 5, num_rows * 5))

    for i, pc in enumerate(point_clouds, start=1):

        ax = fig.add_subplot(num_rows, num_cols, i, projection='3d')

        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.5, c=colors[i-1])

        ax.set_title(titles[i-1])  # Set the title for each subplot

        ax.set_xlabel('X')

        ax.set_ylabel('Y')

        ax.set_zlabel('Z')

        ax.set_xlim([-5, 5])

        ax.set_ylim([-5, 5])

        ax.set_zlim([0, 5])

    plt.tight_layout()

    plt.show()



def display_meshes(meshes, titles, num_cols=3):

    import seaborn as sns

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

        ax.set_xlim([-5, 5])

        ax.set_ylim([-5, 5])

        ax.set_zlim([0, 5])

    plt.tight_layout()

    plt.show()
