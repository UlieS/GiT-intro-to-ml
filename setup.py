from setuptools import find_packages, setup

BASE_DEPENDENCIES = [
    "matplotlib>=3.5.1, <3.6",
    "numpy>=1.22.3, <1.23",
    "streamlit>=0.62.0, <0.63"
    "pandas>=1.4.2, <1.5",
    "seaborn>=0.11.2, <0.12"
]

setup(
    name="ml-intro",
    version="0.1.0",
    author="Ulie Schnaithmann",
    author_email="ulrike.schnaithmann@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9, <3.10",
    install_requires=BASE_DEPENDENCIES,
)
