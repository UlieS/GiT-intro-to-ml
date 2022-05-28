import streamlit as st

from .utils import create_font, display_image


def app():

    create_font(30, '1. Goals of this workshop')
    display_image('ml-meme.png')

    st.markdown('''
            ##
            ##
        ''')

    st.write('''
        - Build intuition for the statistical fundamentals of Machine Learning 
        - Spark interest for the topic and motivation to explore further
        - Debunk the myth that you need a PhD in math to work in Machine Learning         
        - Understand the difference between Machine Learning vs. Statistics vs. Deep Learning vs. Artificial Intelligence
    ''')

    create_font(30, '2. Introductions - Instructor and Attendees')
    st.write("""
        - My name is Ulie and I live in San Francisco
        - This is my first workshop for Girls in Tech
        - 3+ years of experience in Machine Learning, specifically Computer Vision and Natural Language Processing 
        - Bachelor's degree in Psychology + Computer Science (OvG University Magdeburg, Germany)
        - Master's degree in Data and Knowledge Engineering (OvG University Magdeburg, Germany)
        - Hobbies: running (training for the SF marathon right now!), baking bread, traveling
    """)

    create_font(30, "What's your name and your motivation for attending this workshop today?")
