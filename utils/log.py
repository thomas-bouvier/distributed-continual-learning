import json
import os
import logging
import pandas as pd

from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Div
from itertools import cycle

def setup_logging(log_file='log.txt', dummy=False):
    if dummy:
        logging.getLogger('dummy')
        return

    file_mode = 'a' if os.path.isfile(log_file) else 'w'

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    fileout = logging.FileHandler(log_file, mode=file_mode)
    fileout.setLevel(logging.DEBUG)
    fileout.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fileout)

def plot_figure(data, x, y, title=None, xlabel=None, ylabel=None, legend=None,
                x_axis_type='linear', y_axis_type='linear',
                width=800, height=400, line_width=2,
                colors=['red', 'green', 'blue', 'orange',
                        'black', 'purple', 'brown'],
                tools='pan,box_zoom,wheel_zoom,box_select,hover,reset,save',
                append_figure=None):
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
        f = figure(title=title, tools=tools,
                   width=width, height=height,
                   x_axis_label=xlabel or x,
                   y_axis_label=ylabel or '',
                   x_axis_type=x_axis_type,
                   y_axis_type=y_axis_type)
    colors = cycle(colors)
    for i, yi in enumerate(y):
        f.line(data[x], data[yi],
               line_width=line_width,
               line_color=next(colors), legend_label=legend[i])
    f.legend.click_policy = "hide"
    return f

class ResultsLog(object):

    supported_data_formats = ['csv', 'json']

    def __init__(self, path='', title='', params=None, resume=False, data_format='json'):
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
        if data_format not in ResultsLog.supported_data_formats:
            raise ValueError('data_format must of the following: ' +
                             '|'.join(['{}'.format(k) for k in ResultsLog.supported_data_formats]))

        if data_format == 'json':
            self.data_path = '{}.json'.format(path)
        else:
            self.data_path = '{}.csv'.format(path)

        if params is not None:
            filename = '{}.json'.format(path)
            with open(filename, 'w') as fp:
                json.dump(dict(self.args._get_kwargs()), fp, sort_keys=True, indent=4)

        self.plot_path = '{}.html'.format(path)
        self.results = None
        self.first_save = True
        self.clear()

        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
                self.first_save = False
            else:
                os.remove(self.data_path)
                self.results = pd.DataFrame()
        else:
            self.results = pd.DataFrame()

        self.title = title
        self.data_format = data_format

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.results = self.results.append(df, ignore_index=True)

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
        Parameters
        ----------
        title: string
            title of the HTML file
        """
        title = title or self.title
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            if self.first_save:
                self.first_save = False
                print('Plot file saved at: {}'.format(
                    os.path.abspath(self.plot_path)))

            output_file(self.plot_path, title=title)
            plot = column(
                Div(text='<h1 align="center">{}</h1>'.format(title)), *self.figures)
            save(plot)
            self.clear()

        if self.data_format == 'json':
            self.results.to_json(self.data_path, orient='records', lines=True)
        else:
            self.results.to_csv(self.data_path, index=False, index_label=False)