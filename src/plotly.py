# Dependencies
from .msa import MSA
import plotly.graph_objects as go
import numpy as np


# def get_logo(msa, mask=None):
#     # Define all amino acids
#     aa = MSA.ALL_RESIDUES
#     # Get shape of the alignment
#     _, m = msa.shape
#     # Get amino acid frequencies
#     frequencies = msa.get_frequencies()
#     # Get rank for each column in logo, excluding gaps
#     index = np.argsort(-logo[:-1], axis=0)


def plot_occupancy(msa):
    # Get alignment shape
    _, m = msa.shape
    # Define horizontal axis values
    x = np.arange(1, m + 1)
    # Define vertical axis values
    y = msa.get_occupancy()
    # Return trace for occupancy
    return go.Scatter(x=x, y=y, mode='lines+markers', name='Occupancy')


def plot_kl_divergence(msa):
    # Get alignment shape
    _, m = msa.shape
    # Define horizontal axis values
    x = np.arange(1, m + 1)
    # Define vertical axis values
    y = msa.get_kl_divergence()
    # Return trace for occupancy
    return go.Scatter(x=x, y=y, mode='lines+markers', name='K-L divergence')


def plot_shannon_entropy(msa):
    # Get alignment shape
    _, m = msa.shape
    # Define horizontal axis values
    x = np.arange(1, m + 1)
    # Define vertical axis values
    y = msa.get_shannon_entropy()
    # return trace for occupancy
    return go.Scatter(x=x, y=y, mode='lines+markers', name='Shannon\'s entropy')


def plot_lip_consensus(msa):
    raise NotImplementedError


def plot_disorder_consensus(msa):
    raise NotImplementedError


if __name__ == '__main__':
    # TODO Main
    raise NotImplementedError