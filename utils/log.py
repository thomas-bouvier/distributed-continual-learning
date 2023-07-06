import json
import os
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shutil
import torch
import wandb

from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div
from itertools import cycle


def setup_logging(log_file="log.txt", level="info", dummy=False):
    if dummy:
        logging.getLogger("dummy")
        return

    file_mode = "a" if os.path.isfile(log_file) else "w"

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fileout = logging.FileHandler(log_file, mode=file_mode)
    fileout.setLevel(logging.DEBUG)
    fileout.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger("root").addHandler(fileout)


def get_logging_level():
    return logging.getLevelName(logging.getLogger("root").getEffectiveLevel()).lower()


def plot_figure(
    data,
    x,
    y,
    title=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    x_axis_type="linear",
    y_axis_type="linear",
    width=800,
    height=400,
    line_width=2,
    colors=["red", "green", "blue", "orange", "black", "purple", "brown"],
    tools="pan,box_zoom,wheel_zoom,box_select,hover,reset,save",
    append_figure=None,
):
    """
    creates a new plot figures
    example:
        plot_figure(x='epoch', y=['train_loss', 'val_loss'],
                        'title='Loss', 'ylabel'='loss')
    """
    if not isinstance(y, list):
        y = [y]
    xlabel = xlabel or x
    legend = legend or y
    assert len(legend) == len(y)
    if append_figure is not None:
        f = append_figure
    else:
        f = figure(
            title=title,
            tools=tools,
            width=width,
            height=height,
            x_axis_label=xlabel or x,
            y_axis_label=ylabel or "",
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
        )
    colors = cycle(colors)
    for i, yi in enumerate(y):
        f.line(
            data[x],
            data[yi],
            line_width=line_width,
            line_color=next(colors),
            legend_label=legend[i],
        )
    f.legend.click_policy = "hide"
    return f


class ResultsLog:
    supported_data_formats = ["csv", "json"]

    def __init__(
        self, save_path, title="", params=None, data_format="json", dummy=False
    ):
        """
        Parameters
        ----------
        path: string
            path to directory to save data files
        plot_path: string
            path to directory to save plot files
        title: string
            title of HTML file
        params: Namespace
            optionally save parameters for results
        resume: bool
            resume previous logging
        data_format: str('csv'|'json')
            which file format to use to save the data
        """
        self.results = pd.DataFrame()
        self.first_save = True
        self.title = title
        self.data_format = data_format
        self.dummy = dummy
        self.clear()

        if data_format not in self.supported_data_formats:
            raise ValueError(
                "data_format must of the following: "
                + "|".join(["{}".format(k) for k in ResultsLog.supported_data_formats])
            )

        if self.dummy:
            return

        if data_format == "json":
            self.data_path = f"{save_path}.json"
        else:
            self.data_path = f"{save_path}.csv"

        if params is not None:
            filename = f"{save_path}.json"
            with open(filename, "w") as fp:
                json.dump(dict(self.args._get_kwargs()), fp, sort_keys=True, indent=4)
        self.plot_path = f"{save_path}.html"

        if os.path.isfile(self.data_path):
            os.remove(self.data_path)

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        # logging.debug(f"--add: adding content")
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        # logging.debug(f"df.shape: {df.shape}")
        self.results = pd.concat([self.results, df])
        # logging.debug(f"self.results.shape: {self.results.shape}")

    def clear(self):
        self.figures = []

    def plot(self, *kargs, **kwargs):
        """
        add a new plot to the HTML file
        example:
            results.plot(x='epoch', y=['train_loss', 'val_loss'],
                         'title='Loss', 'ylabel'='loss')
        """
        f = plot_figure(self.results, *kargs, **kwargs)
        self.figures.append(f)

    def save(self, title=None):
        """save the json file.

        title: string
            title of the HTML file
        """
        if self.dummy:
            self.clear()
            return

        title = title or self.title
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            if self.first_save:
                self.first_save = False
                print("Plot file saved at: {}".format(os.path.abspath(self.plot_path)))

            output_file(self.plot_path, title=title)
            plot = column(
                Div(text='<h1 align="center">{}</h1>'.format(title)),
                *self.figures,
            )
            save(plot)
            self.clear()

        # https://stackoverflow.com/questions/49545985/catch-pandas-df-to-json-exception
        if self.data_format == "json":
            # logging.debug(f"--save:")
            # logging.debug(f"self.results: {self.results}")
            # Sometimes the following doesn't work for some reason, probably
            # a memory issue.
            # self.results.to_json(self.data_path, orient='records')
            # Replacing that function as suggested on SO
            holder_dictionary = self.results.to_dict(orient="records")
            with open(self.data_path, "w") as outfile:
                json.dump(holder_dictionary, outfile)
                wandb.save(self.data_path)
        else:
            self.results.to_csv(self.data_path, index=False, index_label=False)


class PerformanceResultsLog:
    perf_metrics = {}

    def add(self, batch, value, key=None):
        if batch not in self.perf_metrics:
            self.perf_metrics[batch] = {}

        if key is not None:
            self.perf_metrics[batch] |= {key: value}
        elif isinstance(value, list):
            for i in range(len(value)):
                self.perf_metrics[batch] |= {i: value[i]}

    def get(self, batch):
        return self.perf_metrics[batch]


def plot_representatives(rep_values, rep_labels, num_cols):
    """
    This function plots representative images.

    Args:
        rep_values (list): The representative images.
        rep_labels (list): The labels for the representative images.
        num_cols (int): The number of columns to display.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    num_candidates = len(rep_values)
    fig, ax = plt.subplots(num_candidates // num_cols, num_cols)
    for j in range(num_candidates // num_cols):
        for k in range(num_cols):
            ax[j, k].imshow(rep_values[j * num_cols + k].T, interpolation="none")
            ax[j, k].set_title(rep_labels[j * num_cols + k].item())
            ax[j, k].axis("off")
    return fig


def display(filename, outputs, columns=2, captions=None):
    if outputs.numel() == 0:
        return
    rows = int(math.ceil(len(outputs) / columns))
    fig = plt.figure()
    fig.set_size_inches(16, 6 * rows)
    gs = gridspec.GridSpec(rows, columns)
    row = 0
    col = 0
    for i in range(len(outputs)):
        plt.subplot(gs[i])
        plt.axis("off")
        if captions is not None:
            plt.title(captions[i])
        plt.imshow(np.array(outputs[i].permute(1, 2, 0).cpu()))
    path = f"{filename}.jpg"
    plt.savefig(path)
    plt.close()
    wandb.save(path)
