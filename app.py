import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import *

option = st.sidebar.selectbox(
    'Choose dataset',
    ['boston', 'iris', 'diabetes']
)

if option == 'boston':
    st.title("Boston housing dataset")

    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['target'] = pd.Series(boston.target)

    st.header("Raw data")
    st.write(df)

    st.header("CRIM Histogram")
    st.bar_chart(np.histogram(df['CRIM'])[0])
elif option == 'iris':
    st.title("Iris flower dataset")

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = pd.Series(iris.target)

    st.header('Raw data')
    st.write(df)
elif option == 'diabetes':
    st.title("Diabetes dataset")

    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = pd.Series(diabetes.target)

    st.header('Raw data')
    st.write(df)
else:
    pass
