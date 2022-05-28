from dataclasses import asdict, dataclass

import pandas as pd
import numpy as np


def create_data_matrix():
    return pd.DataFrame(data=[
        asdict(Case('CuteAnimalsFanPage', 'Y', 'Y', 22, 2032, 120)),
        asdict(Case('Doggo_lover24', 'N', 'Y', 18, 1230, 344)),
        asdict(Case('Cute_cat2', 'Y', None, 14, 843, 112)),
        asdict(Case('Influencer101', None, 'Y', 27, 10000, 201))
    ]
    )#, columns=['Username', 'Likes cats', 'Likes dogs', 'Age', '# of followers', 'Avg. daily minutes spent on Instagram'])


def create_level_of_measurement_overview():
    return pd.DataFrame(data=[
        ['Nominal', 'Grouping, No order/direction', 'Color, Categories'],
        ['Ordinal', 'Order, Ranking', 'Education level, Service Quality Rating'],
        ['Interval', 'Distance between measurements is meaningful, but no true zero', 'Temperature in Fahrenheit'],
        ['Ratio', 'Absolute zero is meaningful', 'Weight, Height']
    ], columns=['Level of Measurement', 'Properties', 'Example'])


def generate_boxplot_data():

    # fake up some data
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low))
    return data


def generate_regression_data(variation: int = 20, n: int = 20):
    np.random.seed(2)

    def f(x):
        return x+np.random.randint(-variation**2, variation**2)

    data = [f(x) for x in range(n)]
    return data


@dataclass
class Case:
    username: str
    likes_cats: str
    likes_dogs: str
    age: int
    number_of_followers: int
    daily_minutes_spent_on_instagram: int

