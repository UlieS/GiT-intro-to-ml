
import streamlit as st

from .utils import create_font, display_image


def app():

    create_font(30, 'Fundamentals of Deep Learning - Multi Layer Perceptron')

    st.write('''
    - First applications of "neurons" as logic gates
    - No learning possible (as there a no learnable weights), merely settable inputs
    ''')
    display_image('logic-gates.png')

    st.write('''
    - Extension into Perceptron algorithm - input is combined into *weighted* sum 
    - Weights and thresholds are learnable → Perceptron can learn decision boundary to fit to data 
    - Weights are randomly initialized and are adjusted during optimization (using Stochastic Gradient Descent)
    - In its most basic design, a perceptron can be regarded as a single neuron → still linear model
    - Linearity constraint has limitations → *Multi Layer Perceptron* to separate non-linear data 
    - → Stacking of layers which feed information from previous layers forward to the next 
    ''')

    display_image('mlp.png')


