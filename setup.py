from setuptools import setup, find_packages

setup(
    name="churn-prediction-app",
    version="1.0.0",
    description="Customer Churn Prediction Web Application",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.1.3",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.2",
        "plotly>=5.17.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.13.0",
        "joblib>=1.3.2",
    ],
    python_requires=">=3.8",
)