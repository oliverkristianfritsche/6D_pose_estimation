from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt


def plot_angular_errors(df, run_number):
    df_run = df[df['Run'] == run_number]
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    titles = ['Yaw Error', 'Pitch Error', 'Roll Error']
    colors = ['r', 'g', 'b']

    for i, ax in enumerate(axes):
        for index, row in df_run.iterrows():
            angular_errors = row['Angular Error History']
            if isinstance(angular_errors, list) and all(isinstance(e, list) and len(e) == 3 for e in angular_errors):
                # Extract and plot each angular error component
                component_errors = [e[i] for e in angular_errors]
                ax.plot(component_errors, color=colors[i], label=f'Trial {row["Trial"]}')
            else:
                print(f"Data format issue at Run {row['Run']}, Trial {row['Trial']}")
            ax.set_title(titles[i])
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Degrees')
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()

# Plotting functions for detailed histories
def plot_detailed_histories(loss_histories, angular_error_histories):
    plt.figure(figsize=(15, 7))
    for i in range(len(loss_histories)):
        plt.subplot(2, 1, 1)
        plt.plot(loss_histories[i], label=f'Loss Run {i+1}')
        plt.title('Loss History for Each Optimization Run')
        plt.xlabel('Trial')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(angular_error_histories[i], label=f'Angular Error Run {i+1}')
        plt.title('Angular Error History for Each Optimization Run')
        plt.xlabel('Trial')
        plt.ylabel('Angular Error (degrees)')

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_histograms(best_losses, best_angular_errors):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.hist(best_losses, bins=20, edgecolor='black')
    plt.title('Histogram of Best Losses')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(best_angular_errors, bins=20, edgecolor='black')
    plt.title('Histogram of Best Angular Errors')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()



def plot_results(df, plot_type='Loss History', runs=None, trials=None, figsize=(10, 5), title=None, xlabel='Iteration', ylabel=None):
    """ General plotting for optimization metrics. """
    if runs is not None:
        df = df[df['Run'].isin(runs)]
    if trials is not None:
        df = df[df['Trial'].isin(trials)]

    plt.figure(figsize=figsize)
    if title is None:
        title = f'{plot_type} per Iteration'
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel if ylabel else plot_type)

    for (run, trial), group in df.groupby(['Run', 'Trial']):
        iterations = range(len(group.iloc[0][plot_type]))
        data = [item for sublist in group[plot_type].tolist() for item in sublist]  # Flatten if nested
        plt.plot(iterations, data, label=f'Run {run}, Trial {trial}')
        plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.show()