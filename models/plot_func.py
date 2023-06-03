import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def pieplot(
    df: pd.DataFrame,
    col: str = 'income_50k',
    labels: list = ['<=50K', '>50K'],
    save: str = None
):
    """ Plot a pieplot for the given column.
    df: dataframe
        Dataset to plot.
    col: str
        Column to plot.
    labels: list
        Labels for the pieplot.
    save: str
        Path to save the plot. If None, the plot is not saved.
    """
    df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(5,5), labels=['', ''])
    # keep the original labels
    plt.legend(labels=labels)
    plt.ylabel('')
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def barplot(
    df: pd.DataFrame,
    col: str = 'income_50k',
    labels: list = ['≤50K', '>50K'],
    save: str = None
):
    """ Plot a barplot for the given column.
    df: dataframe
        Dataset to plot.
    col: str
        Column to plot.
    labels: list
        Labels for the barplot.
    save: str
        Path to save the plot. If None, the plot is not saved.
    """
    sns.barplot(x=df[col].value_counts(normalize=True).index, y=df[col].value_counts(normalize=True)*100, palette='Blues_d')
    # keep the original labels
    plt.xticks([0, 1], labels)
    plt.ylabel('Frequency (%)')
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
    

def pieplots(
    df: pd.DataFrame,
    cols: list,
    grid: tuple = (None, 5),
    figsize: tuple = (20, 10),
    save: str = None,
    other_freq: float = 0.05
):
    """ Plot pieplots for the given columns in a grid of 4 columns.
    df: dataframe
        Dataset to plot.
    cols: list
        Columns to plot.
    grid: tuple
        Grid size for the subplots.
    figsize: tuple
        Figure size.
    save: str
        Path to save the plot. If None, the plot is not saved.
    other_freq: float
        Frequency threshold for the 'other' category.
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
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()


def barplots(
    df: pd.DataFrame,
    cols: list,
    grid: tuple = (None, 5),
    figsize: tuple = (20, 10),
    save: str = None,
    other_freq: float = 0.05
):
    """ Plot barplots for the given columns in a grid of 4 columns.
    df: dataframe
        Dataset to plot.
    cols: list
        Columns to plot.
    grid: tuple
        Grid size for the subplots.
    figsize: tuple
        Figure size.
    save: str
        Path to save the plot. If None, the plot is not saved.
    other_freq: float
        Frequency threshold for the 'other' category.
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
        # change the color palette
        counts.plot.bar(ax=subplot, color=sns.color_palette('Blues', len(counts)))

        for item in subplot.get_xticklabels():
            item.set_rotation(0)

        subplot.set_title(variable)

        # change the color of the 'other' category
        if 'other' in counts.index:
            subplot.patches[-1].set_facecolor('grey')
        subplot.set_ylabel('')


    # remove unused graphs
    for i in range(len(cols), grid[0]*grid[1]):
        fig.delaxes(ax.flatten()[i])

    # save the figure
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

# def plot_conf_matrix(y_test, y_pred, savename=None, normalize='true'):
def plot_conf_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    save: str = None,
    show: bool = True,
    normalize: str = 'true'
):
    """ Plot the confusion matrix.
    y_test: np.ndarray
        Test labels.
    y_pred: np.ndarray
        Predicted labels.
    savename: str
        Path to save the plot. If None, the plot is not saved.
    normalize: str
        Normalization mode for the confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    cm = pd.DataFrame(cm, index=['≤50K', '>50K'], columns=['≤50K', '>50K'])
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
