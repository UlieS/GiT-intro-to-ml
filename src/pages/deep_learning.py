
import streamlit as st
import urllib.request
from PIL import Image

from .utils import create_font


def app():

    create_font(30, 'Multi Layer Perceptron - The fundamentals of Deep Learning')

    st.write('''
    Nothing
    ''')

    urllib.request.urlretrieve(
      'https://media.geeksforgeeks.org/wp-content/uploads/20210318103632/gfg-300x300.png',
       "gfg.png")

    img = Image.open("gfg.png")
    st.image(img)


