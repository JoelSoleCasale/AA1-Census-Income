import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def pieplots(df, cols, grid=(None, 5), figsize=(20, 10), savename=None, other_freq=0.05):
    """
    Plot pieplots for the given columns in a grid of 4 columns.
    """
    if grid[0] is None:
        grid = (int(np.ceil(len(cols) / grid[1])), grid[1])
    # create a figure and a grid of subplots
    fig, ax = plt.subplots(grid[0], grid[1], figsize=figsize)
    for variable, subplot in zip(cols, ax.flatten()):
        counts = df[variable].value_counts(normalize=True)
        counts = counts[counts > other_freq]
        if 1 - counts.sum() > 0:
            counts['other'] = 1 - counts.sum()
        counts.plot.pie(ax=subplot, autopct='%1.1f%%')
        subplot.set_title(variable)
        subplot.set_ylabel('')

    # remove unused graphs
    for i in range(len(cols), grid[0]*grid[1]):
        fig.delaxes(ax.flatten()[i])

    # save the figure
    if savename:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()


def plot_conf_matrix(y_test, y_pred, savename=None, normalize='true'):
    # change axis names from 0 and 1 to <=50K and >50K
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    cm = pd.DataFrame(cm, index=['<=50K', '>50K'], columns=['<=50K', '>50K'])
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if savename:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()
