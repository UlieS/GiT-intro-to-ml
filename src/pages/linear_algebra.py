import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from .utils import create_font


def app():
    create_font(30, 'Linear Algebra used in Machine Learning')

    ### Vectors
    create_font(20, "Vectors and Matrices")
    st.write("""
            - Vectors are a convenient and efficient way to compute and work with high dimensional data 
            - can be regarded as "list of values"
            - eg. vector of parameters of regression line in previous section: 
            
            """)
    st.latex(r'''
        \begin{bmatrix}
            b_0  \\
            b_1 
        \end{bmatrix}
        = 
        \begin{bmatrix}
            1 \\
            2 
        \end{bmatrix}
    ''')

    st.markdown("![Alt Text](https://miro.medium.com/max/1400/1*c9WXUJdx00_fniK8h--k1A.gif")
    st.write("""
        - Matrix: array of (row and column) vectors
        - $m x n$ indicates the dimensions of the matrix, below is $2 x 3 $
    """)
    st.latex(r'''
        \begin{bmatrix}
            2 & 3b & 23.1 \\
            19x & 3 & 2  
        \end{bmatrix}
    ''')

    st.write(r'''Basic Matrix Operations:'
             - Addition/Subtraction requires identical shape of matrices
             ''')
    st.latex(r'''
        \begin{bmatrix}
            1 & 2  \\
            3 & 4 
        \end{bmatrix}
        + 
        \begin{bmatrix}
            10 & 20 \\
            30 & 40
        \end{bmatrix}
        = 
        \begin{bmatrix}
            11 & 22 \\
            33 & 44
        \end{bmatrix}
    ''')

    st.write(r'''- Multiplication by a scalar:    
    ''')
    st.latex(r'''
        2
        * 
        \begin{bmatrix}
            10 & 20 \\
            30 & 40
        \end{bmatrix}
        = 
        \begin{bmatrix}
            20 & 40 \\
            60 & 80
        \end{bmatrix}
    ''')

    st.write(r'''Matrix Multiplication:
        - elements of the rows in the first matrix are multiplied with corresponding columns in the second matrix
        - requirement: $ n x m; m x b $ (number of columns of first matrix == number of rows of second matrix)
        - resulting dimensions: $ n x m * m x a -> n x a $
        - matrix multiplication the core operation of Deep Learning
    ''')
    st.markdown("![Alt Text](https://textimgs.s3.amazonaws.com/boundless-algebra/x-multiplication-diagram-2.svg#fixme")

    st.write(r''' Dot product (inner product):
        - a way to multiply vectors with the result being a scalar
        - geometric meaning: 
            - indicates angle between the two vectors
            - gives the degree of correlation between the two vectors 
    ''')

    x_dir = int(st.text_input('X direction:'))
    y_dir = int(st.text_input('Y direction:'))
    if x_dir and y_dir:
        plot_vectors(x_dir, y_dir)
        dot_product = np.dot((1, 1), (x_dir, y_dir))
        cos_sim = dot_product/(np.linalg.norm([1, 1])*np.linalg.norm([x_dir, y_dir]))
        st.write(fr'''
        - Angle between vectors: Dot product = {round(np.degrees(dot_product), 1)}
        - How to interpret this value? Cosine similarity!
        Intuition: 
        1. agnle = 90 degrees (vectors are perpendicular): 
        - correlation between the two vector is 0, vectors are independent to each other
        2. angle = 0 degrees:
        - correlation between the two vector is 1, vectors are totally dependent to each other
        - *correlation between the vectors above: {round(cos_sim, 3)}*
    ''')
    else:
        plot_vectors()


def plot_vectors(x_dir: int = 1, y_dir: int = -1) -> None:
    fig, ax = plt.subplots(1)
    ax.quiver([0, 0], [0, 0], [1, x_dir], [1, y_dir], angles='xy', scale_units='xy', scale=1)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    st.pyplot(fig)
    return

