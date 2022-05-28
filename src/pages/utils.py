import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import urllib


def create_font(font_size: int, text: str, color: str = 'Green') -> None:
    st.markdown("<style>.big-font {font-size:"+str(font_size)+"px !important; color:"+color+"}</style>", unsafe_allow_html=True)
    st.markdown('<p class="big-font">'+text, unsafe_allow_html=True)


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC
    This code is taken from https://github.com/jakevdp/PythonDataScienceHandbook
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def display_image(filename: str):
    urllib.request.urlretrieve(
        f'https://raw.githubusercontent.com/UlieS/GiT-intro-to-ml/main/images/{filename}',
        "img.png")

    img = Image.open("img.png")
    st.image(img)


def plot_3D(X, y, elev=30, azim=30, ):
    r = np.exp(-(X ** 2).sum(1))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    st.pyplot(fig)