# -*- coding: utf-8 -*-
"""
Plotting helpers of the geatpy core (moeaplot, soeaplot, trcplot, varplot), implemented with
matplotlib following the geatpy 2.7.0 documentation. They live in Python because they only drive
matplotlib; the numeric operators are in the Rust extension.
"""
import os

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, ax, label, save_flag, gen, interval, title, save_path, grid_flag, animated):
    plt = _plt()
    if title:
        ax.set_title(title if gen is None else '%s (gen %d)' % (title, gen))
    elif gen is not None:
        ax.set_title('Generation %d' % gen)
    if label or gen is not None:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    ax.grid(bool(grid_flag))
    if save_flag:
        name = (title or 'Plot') + ('' if gen is None else ' %d' % gen) + '.svg'
        fig.savefig(os.path.join(save_path or '', name), dpi=300, bbox_inches='tight')
    if animated:
        plt.pause(0.1 if interval is None else interval)
    return ax


def _points_plot(data, label, save_flag, ax, gen, interval, title, save_path, xyz_label, grid_flag, default_title):
    plt = _plt()
    data = np.atleast_2d(np.asarray(data, dtype=float))
    dim = data.shape[1]
    label = '' if label is None else label
    animated = ax is None or (not isinstance(ax, str) and ax is not None)
    if isinstance(ax, str) or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') if dim == 3 else fig.add_subplot(111)
    else:
        fig = ax.figure
        ax.cla()
    if dim == 2:
        xyz_label = xyz_label or ['F1', 'F2']
        ax.plot(data[:, 0], data[:, 1], 'o', markersize=4, label=label)
        ax.set_xlabel(xyz_label[0])
        ax.set_ylabel(xyz_label[1] if len(xyz_label) > 1 else '')
    elif dim == 3:
        xyz_label = xyz_label or ['F1', 'F2', 'F3']
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=12, label=label)
        ax.set_xlabel(xyz_label[0])
        ax.set_ylabel(xyz_label[1] if len(xyz_label) > 1 else '')
        ax.set_zlabel(xyz_label[2] if len(xyz_label) > 2 else '')
    else:
        xyz_label = xyz_label or ['Dimension Number', 'Value']
        x = np.arange(1, dim + 1)
        for i, row in enumerate(data):
            ax.plot(x, row, '-', linewidth=0.8, color='tab:blue', label=label if i == 0 else None)
        ax.set_xlabel(xyz_label[0])
        ax.set_ylabel(xyz_label[1] if len(xyz_label) > 1 else '')
    return _finish(fig, ax, label, save_flag, gen, interval, title or default_title, save_path, grid_flag, animated)


def moeaplot(ObjV, Label=None, saveFlag=False, ax='static', gen=None, interval=None, title=None, save_path=None,
             xyzLabel=None, gridFlag=False):
    """Plot objective values: points for 2 or 3 objectives, one line per individual otherwise.

    Pass ax=None to start an animation and the returned axes on later calls."""
    return _points_plot(ObjV, Label, saveFlag, ax, gen, interval, title, save_path, xyzLabel, gridFlag,
                        'Pareto Front Plot')


def varplot(Vars, Label=None, saveFlag=False, ax='static', gen=None, interval=None, title=None, save_path=None,
            xyzLabel=None, gridFlag=False):
    """Plot decision variables: points for 2 or 3 variables, one line per individual otherwise."""
    return _points_plot(Vars, Label, saveFlag, ax, gen, interval, title, save_path, xyzLabel, gridFlag,
                        'Variables Value Plot')


def soeaplot(ValueSet, Label=None, saveFlag=False, ax='static', gen=None, interval=None, title=None, save_path=None,
             xlabel=None, ylabel=None, gridFlag=False):
    """Plot a value (e.g. the best objective) against the generation number."""
    plt = _plt()
    values = np.asarray(ValueSet, dtype=float).reshape(-1)
    animated = ax is None or not isinstance(ax, str)
    if isinstance(ax, str) or ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        ax.cla()
    ax.plot(np.arange(len(values)), values, '-', label='' if Label is None else Label)
    ax.set_xlabel(xlabel or 'Number of Generation')
    ax.set_ylabel(ylabel or 'Value')
    return _finish(fig, ax, Label, saveFlag, gen, interval, title, save_path, gridFlag, animated)


def trcplot(trace, labels, titles=None, save_path=None, xlabels=None, ylabels=None, gridFlags=None):
    """Plot the columns of an evolution trace. labels is a list of label lists; each inner list makes one
    figure using the next columns of trace."""
    plt = _plt()
    trace = np.atleast_2d(np.asarray(trace, dtype=float))
    if trace.shape[0] == 1 and trace.shape[1] > 1 and len(labels) == 1 and len(labels[0]) == 1:
        trace = trace.T
    col = 0
    axes = []
    for i, group in enumerate(labels):
        fig, ax = plt.subplots()
        for name in group:
            ax.plot(np.arange(trace.shape[0]), trace[:, col], label=name)
            col += 1
        ax.legend()
        ax.set_xlabel(xlabels[i] if xlabels else 'Number of Generation')
        ax.set_ylabel(ylabels[i] if ylabels else 'Value')
        ax.grid(bool(gridFlags[i]) if gridFlags else False)
        if titles:
            ax.set_title(titles[i])
            if save_path is not None:
                fig.savefig(os.path.join(save_path, titles[i] + '.svg'), dpi=300, bbox_inches='tight')
        axes.append(ax)
    plt.show()
    return axes
