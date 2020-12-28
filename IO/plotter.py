"""
Author: Rex Geng

API for:

- plotting single/multiple trace(s) data
- confusion matrices
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

rc('text', usetex=True)


class PlotData:
    def __init__(self, x, y, marker_shape, plot_name, color='k'):
        self.name = plot_name
        self.x = x
        self.y = y
        self.shape = marker_shape
        self.color = color

    def plot(self, *args, **kwargs):
        plot_multiple_traces(self, *args, **kwargs)


class PlotDataSequence:
    def __init__(self, plot_data_sequence):
        self.sequence = plot_data_sequence

    def add_plot_data_sequence(self, plot_data_sequence):
        if isinstance(plot_data_sequence, PlotDataSequence):
            self.sequence += plot_data_sequence.sequence
        else:
            self.sequence += plot_data_sequence

    def __len__(self):
        return len(self.sequence)

    def get_xs(self):
        return [p.x for p in self.sequence]

    def get_ys(self):
        return [p.y for p in self.sequence]

    def get_names(self):
        return [p.name for p in self.sequence]

    def get_colors(self):
        return [p.color for p in self.sequence]

    def get_shapes(self):
        return [p.shape for p in self.sequence]

    def plot(self, *args, **kwargs):
        plot_multiple_traces(self, *args, **kwargs)


def plot_multiple_traces(
        data, size=(600, 12), figsize=(20, 10), xtitle=None, ytitle=None, title=None,
        use_seasborn=False, save_to_file=True, file_name='plot.png',
        log_x=False, log_y=False, use_legend=True, title_font=30, axis_label_font=30,
        axis_tick_font=20, legend_font=15, save_to_pdf=True, fig_transparent=False, use_scatter=(False,),
        xlim=None, ylim=None, customize_legend=None,
):
    if use_seasborn:
        sns.set()

    colors, names, shapes, xs, ys = unpack_plot_data(data)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for idx, (x, y, shape, name, color) in enumerate(zip(xs, ys, shapes, names, colors)):
        if use_scatter[idx]:
            ax.scatter(x, y, marker=shape, s=size[0], label=name, color=color)
        else:
            ax.plot(x, y, marker=shape, markersize=size[1], label=name, color=color)

    ax.tick_params(axis="x", labelsize=axis_tick_font)
    ax.tick_params(axis="y", labelsize=axis_tick_font)
    ax.grid()

    if log_y:
        ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')
    if xtitle is not None:
        ax.set_xlabel(xtitle, fontsize=axis_label_font)
    if ytitle is not None:
        ax.set_ylabel(ytitle, fontsize=axis_label_font)
    if title is not None:
        ax.set_title(title, fontsize=title_font)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if use_legend:
        if customize_legend is not None:
            if len(customize_legend[0]) == 0:
                ax.legend(fontsize=legend_font, loc=customize_legend[1])
            else:
                ax.legend(handles=customize_legend[0], fontsize=legend_font, loc=customize_legend[1])
        else:
            ax.legend(fontsize=legend_font)

    # save the figure
    if not save_to_file:
        plt.show()
    else:
        if save_to_pdf:
            plt.savefig(file_name.replace('png', 'pdf'), format='pdf', transparent=fig_transparent, bbox_inches='tight')

        plt.savefig(file_name, format='png', bbox_inches='tight')


def unpack_plot_data(data):
    if isinstance(data, PlotDataSequence):  # PlotDataSequnece Type
        xs = data.get_xs()
        ys = data.get_ys()
        names = data.get_names()
        shapes = data.get_shapes()
        colors = data.get_colors()

    elif isinstance(data, PlotData):  # PlotData Type
        x, y, name, shape, color = data.x, data.y, data.name, data.shape, data.color
        xs, ys, names, shapes, colors = [x], [y], [name], [shape], [color]
    else:
        if len(data[0]) > 1:  # Tuple for multiple curves
            xs, ys, names, shapes, colors = data
        else:  # Tuple for single curve
            x, y, name, shape, color = data
            xs, ys, names, shapes, colors = [x], [y], [name], [shape], [color]
    return colors, names, shapes, xs, ys


def plot_confusion_matrix(cm, title='Confusion matrix', normalize=False, cmap=plt.cm.Blues, save_path=None, pdf=False,
                          nclass=6):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(nclass)
    plt.xticks(tick_marks, np.arange(nclass), rotation=45)
    plt.yticks(tick_marks, np.arange(nclass))  # todo latter adding class text annotations

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path is not None:
        if pdf:
            plt.savefig(save_path.replace('png', 'pdf'), format='pdf', transparent=False, bbox_inches='tight')
        plt.savefig(save_path, format='pdf', transparent=False, bbox_inches='tight')
    else:
        plt.show()
