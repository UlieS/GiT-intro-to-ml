import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from .data import create_data_matrix, generate_boxplot_data, generate_regression_data, create_level_of_measurement_overview
from .utils import create_font


def app():

    descriptive_statistics_section()
    with st.expander("See data"):
        st.write('''
        - Let's look at some small scale sample data! 
        - e.g. Survey of instagram users:
        ''')
        data = create_data_matrix()
        st.table(data)

    levels_of_measurement_section()
    variability_section()
    distribution_section()
    regression_section()


def levels_of_measurement_section():
    st.markdown('''
        ##
    ''')
    create_font(20, 'How to look at data?')
    st.write("""
        - Data needs to be interpreted in the terms of their **levels of measurement**:
        - This helps choose the best methods to analyze/summarize data
    """)

    data = create_level_of_measurement_overview()
    st.table(data)


def regression_section():
    st.markdown('''
        ---
        ##
    ''')

    create_font(20, 'Linear Regression')

    st.write("""
        - Summarize and analyze relationships between two continuous variables
        - X: predictor, explanatory, independent variable
        - Y: response, outcome, dependent variable
    """)
    ### widgets
    variation = st.slider('Variation', 1, 10, 10)
    n = st.slider('Number of cases', 1, 20, 200)

    fig = plt.figure(figsize=(5, 5))
    axes = fig.add_axes([0,0,1,1,])

    data = generate_regression_data(variation, n)
    x = range(len(data))
    axes.scatter(x, data)
    axes.set_xlim([0,200])
    axes.set_ylim([-100,200])

    estimated_intercept = st.text_input('Intercept (b0)')
    estimated_slope = st.text_input('Slope (b1)')
    if estimated_intercept != '' and estimated_slope != '':
        estimated_intercept = float(estimated_intercept)
        estimated_slope = float(estimated_slope)
        estimated_function = lambda x: estimated_intercept + estimated_slope*x
        estimated_ys = [estimated_function(val) for val in range(len(data))]
        # draw guessed line
        plt.plot(x, estimated_ys)

    st.pyplot(fig)

    st.write(f"""
                - estimated regression function to describe this relationship: 
                    $f(x) = b_0 +b_1x$
                with $b_0 - intercept, b_1 - slope$ 
                - calculate the weights by minimizing the distance from each data point to the line 
                (e.g. method of ordinary least squares)
            """)

    x = np.array(x).reshape(-1,1)
    y = np.array(data)

    with st.expander('Calculation in Python'):
        with st.echo():
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(x,y)
            st.write(f"""Intercept: {round(model.intercept_, 2)}/n, Slope: {round(model.coef_[0],2)}/n, '
                        Coefficient of Determination (R^2): {round(model.score(x, y), 2)}/n""")

        st.write(f""" 
                    - Evaluation of goodness of fit: $R^2$ - How well does the equation describe the data?
                    - $R^2$ measures the explained variance of two variables (i.e. to what extent can the behavior of 
                    the response be explained by the dependant? How much is left to explain by other unknown variables?)
                    - Assuming a good enough fit: plug-in in to original formula:'
                    $f(x) = {round(model.intercept_, 2)} +{round(model.coef_[0],2)}x$'
                    'Use to make predictions!
                """)


def variability_section():
    st.markdown('''
        ##
        ##
    ''')

    create_font(20, "Variability of data")
    st.write("""
        - Another way of describing data: distributions, showing the variability/dispersion of data
        - Measures of central tendency: Mode, Median, Mean
        - Example: 
    """)
    create_font(40, '[1, 3, 4, 6, 6, 7, 8]')
    create_font(20, ' --- Median (Middle): 6')
    create_font(20, ' --- Mean (Average): 5')
    create_font(20, ' --- Mode (Most Common): 6')

    st.markdown('''
        ##
        ##
    ''')
    st.write("""
        - Standard Deviation/Variance: measures to gauge how far away a data point is from the center on average
        - e.g Boxplot:
    """)
    data = generate_boxplot_data()
    with st.expander('Box Plot Sample'):
        fig, ax1 = plt.subplots(figsize=(2,2))
        ax1.boxplot(data)
        st.pyplot(fig)


def descriptive_statistics_section():
    create_font(40, 'Descriptive Statistics')
    st.write("Why do we need Statistics?")
    st.write("""
        - to think about data in a structured/organized way
        - to draw conclusions about the dataset or make predictions about its future
        - Cases: individual data points
        - Variables: attributes that each data point takes
    """)


def distribution_section():
    st.markdown('''
        ##
        ##
    ''')

    create_font(40, 'What is a distribution?')

    st.write(''' -> function which shows all the possible values data could take + their frequency
             - helps describe properties of data and the relationship between observations
             - makes future observations more predictable
             - ''')
    st.write('''How do we describe distributions?
            - key parameters, eg. normal distribution: sigma (variation) and mu (average)
            - data will fall within the range specified by these parameters with a certain probability 
            - more data points -> more refined "shape" of empirical distributions
            ''')

    create_font(25, 'Example: Normal Distribution (Gaussian)')

    st.text('Empiric observations of different heights in cm')
    number_of_heights = st.slider('Select number of people (n) observed', 0, 100, 5)
    type_of_distribution = st.selectbox(label='Type of distribution', options=['Normal', 'Uniform', 'Chi-Square'])
    sigma = st.slider('Select variety of heights (sigma)', 1, 10, 10)
    show_histogram = st.checkbox(label='Show Histogram')

    # histogram
    if type_of_distribution == 'Normal':
        heights = np.random.normal(loc=160, scale=sigma, size=number_of_heights)
    elif type_of_distribution == 'Uniform':
        heights = np.random.uniform(low=150, high=200, size=number_of_heights)

    elif type_of_distribution == 'Chi-Square':
        degrees_of_freedom = st.slider('Select degrees of freedom (Chi-Square only)', 0, 100, 5)
        heights = np.random.chisquare(df=degrees_of_freedom, size=number_of_heights)

    constant_x = [1] * len(heights)

    fig, ax = plt.subplots(2)

    if not show_histogram:
        ax[0].scatter(heights, constant_x)
    if show_histogram:
        ax[1].hist(heights)

        st.pyplot(fig)