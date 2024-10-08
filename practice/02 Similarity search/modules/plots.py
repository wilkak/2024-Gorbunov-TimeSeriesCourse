import numpy as np
import pandas as pd

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)


def plot_ts_set(ts_set: np.ndarray, title: str = 'Input Time Series Set') -> None:
    """
    Plot the time series set

    Parameters
    ----------
    ts_set: time series set
    title: title of plot
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title=title,
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="browser")



def mplot2d(x: np.ndarray, y: np.ndarray, plot_title: str = None, x_title: str = None, y_title: str = None, trace_titles: np.ndarray = None) -> None:
    """
    Multiple 2D Plots on figure for different experiments

    Parameters
    ----------
    x: values of x axis of plot
    y: values of y axis of plot
    plot_title: title of plot
    x_title: title of x axis of plot
    y_title: title of y axis of plot
    trace_titles: titles of plot traces (lines)
    """

    fig = go.Figure()

    for i in range(y.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=y[i], line=dict(width=3), name=trace_titles[i]))

    fig.update_xaxes(showgrid=False,
                     title=x_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2,
                     tickvals=x)
    fig.update_yaxes(showgrid=False,
                     title=y_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black')),
                      width=1000,
                      height=600
                      )

    fig.show(renderer="browser")



def plot_bestmatch_data(ts: np.ndarray, query: np.ndarray) -> None:
    """
    Visualize the input data (time series and query) for the best match task

    Parameters
    ----------
    ts: time series
    query: query
    """

    query_len = query.shape[0]
    ts_len = ts.shape[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.1, 0.9], subplot_titles=("Query", "Time Series"), horizontal_spacing=0.04)

    fig.add_trace(go.Scatter(x=np.arange(query_len), y=query, line=dict(color=px.colors.qualitative.Plotly[1])),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color=px.colors.qualitative.Plotly[0])),
                row=1, col=2)

    fig.update_annotations(font=dict(size=24, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      title_x=0.5)

    fig.show(renderer="browser")




def plot_bestmatch_results(ts: np.ndarray, query: np.ndarray, bestmatch_results: dict) -> None:
    """
    Visualize the best match results by overlaying found subsequences on the time series.

    Parameters
    ----------
    ts: time series (np.ndarray)
    query: query (np.ndarray)
    bestmatch_results: output data found by the best match algorithm (dict)
    """
    query_len = query.shape[0]
    ts_len = ts.shape[0]

    # Извлечение индексов и расстояний лучших совпадений
    best_match_indices = bestmatch_results['indices']
    best_match_distances = bestmatch_results['distances']

    # Создаем subplot для отображения временного ряда и лучших совпадений
    fig = make_subplots(
        rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=("Time Series with Best Matches")
    )

    # Визуализация временного ряда
    fig.add_trace(
        go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color='blue'), name="Time Series"),
        row=1, col=1
    )

    # Цвета для отображения лучших совпадений
    colors = ['green', 'orange', 'purple']

    # Визуализация лучших совпадений на графике временного ряда
    for i, best_match_index in enumerate(best_match_indices):
        fig.add_trace(
            go.Scatter(
                x=np.arange(best_match_index, best_match_index + query_len),
                y=ts[best_match_index:best_match_index + query_len],
                line=dict(color=colors[i % len(colors)], width=4, dash='dash'),
                name=f"Best Match {i+1} (Dist: {best_match_distances[i]:.3f})"
            ),
            row=1, col=1
        )

    # Настройка осей и заголовков
    fig.update_layout(
        title="Best Match Results Overlay on Time Series", height=600,
        xaxis_title="Time", yaxis_title="Amplitude"
    )

    # Обновляем оси
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    # Отображаем график
    fig.show()




def pie_chart(labels: np.ndarray, values: np.ndarray, plot_title='Pie chart') -> None:
    """
    Build the pie chart

    Parameters
    ----------
    labels: sector labels
    values: values
    """

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_traces(textfont_size=20)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      legend=dict(font=dict(size=20, color='black')),
                      width=700,
                      height=500
                      )

    fig.show(renderer="browser")

