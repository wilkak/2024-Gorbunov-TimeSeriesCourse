import numpy as np
import datetime

import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from modules.mp import *

from typing import Tuple

def heads_tails(consumptions: dict, cutoff, house_idx: list) -> Tuple[dict, dict]:

    """
    Split time series into two parts: Head and Tail

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses

    Returns
    --------
    heads: heads of time series
    tails: tails of time series
    """

    heads, tails = {}, {}
    for i in house_idx:
        heads[f'H_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index < cutoff]
        tails[f'T_{i}'] = consumptions[f'House{i}'][consumptions[f'House{i}'].index >= cutoff]
    
    return heads, tails


def meter_swapping_detection(heads, tails, house_idx, m):
    """
    Find the swapped time series pair

    Parameters
    ---------
    heads: heads of time series
    tails: tails of time series
    house_idx: indices of houses
    m: subsequence length

    Returns
    --------
    min_score: time series pair with minimum swap-score
    """
    min_score = float('inf')
    best_pair = {'i': None, 'j': None, 'mp_j': None}
    eps = 1e-8  
    for i in house_idx:
        for j in house_idx:
            if i != j:
               
                head_i = heads[f'H_{i}'].values.flatten()
                tail_j = tails[f'T_{j}'].values.flatten()
                tail_i = tails[f'T_{i}'].values.flatten()

                mp_ij = compute_mp(ts1=head_i, m=m, ts2=tail_j)
                mp_ii = compute_mp(ts1=head_i, m=m, ts2=tail_i)

                score = np.min(mp_ij['mp']) / (np.min(mp_ii['mp']) + eps)

                if score < min_score:
                    min_score = score
                    best_pair['i'] = i
                    best_pair['j'] = j
                    best_pair['mp_j'] = mp_ij

    return best_pair

def plot_consumptions_ts(consumptions: dict, cutoff, house_idx: list):
    """
    Plot a set of input time series and cutoff vertical line

    Parameters
    ---------
    consumptions: set of time series
    cutoff: pandas.Timestamp
        Cut-off point
    house_idx: indices of houses
    """

    num_ts = len(consumptions)

    fig = make_subplots(rows=num_ts, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)

    for i in range(num_ts):
        fig.add_trace(go.Scatter(x=list(consumptions.values())[i].index, y=list(consumptions.values())[i].iloc[:,0], name=f"House {house_idx[i]}"), row=i+1, col=1)
        fig.add_vline(x=cutoff, line_width=3, line_dash="dash", line_color="red",  row=i+1, col=1)

    fig.update_annotations(font=dict(size=22, color='black'))
    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title='Houses Consumptions',
                      title_x=0.5,
                      title_font=dict(size=26, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)', 
                      height=800,
                      legend=dict(font=dict(size=20, color='black'))
                      )

    fig.show(renderer="browser")
