
import streamlit as st
from pages import basic_statistics, introduction, linear_algebra, machine_learning, deep_learning
from multipage import MultiPage
# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("Girls in Tech - A Visual introduction to Machine Learning")

# Add all your application here

app.add_page("Introduction", introduction.app)
app.add_page("Basic Statistics", basic_statistics.app)
app.add_page("Linear Algebra", linear_algebra.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Deep Learning", deep_learning.app)

# The main app
app.run()