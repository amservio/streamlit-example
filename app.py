import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

    CRIM_graph = st.sidebar.checkbox("CRIM Histogram")
    CHAS_graph = st.sidebar.checkbox("Charles River pricing effect")
    INDUS_graph = st.sidebar.checkbox("Age x Industrialization")
    NOX_graph = st.sidebar.checkbox("Pupil/teacher x NOX regression")

    st.header("Raw data")
    st.write(df)

    sns.set_style("darkgrid")

    if CRIM_graph:
        st.header("CRIM Histogram")
        g = sns.displot(df, x='CRIM')
        st.pyplot(g)

    if CHAS_graph:
        st.header("Charles River pricing effect")
        g = sns.catplot(data=df, x='CHAS', y='target', kind = 'swarm')
        st.pyplot(g)

    if INDUS_graph:
        st.header("Age x Industrialization")
        g = sns.relplot(data=df, x='AGE', y='INDUS')
        st.pyplot(g)

    if NOX_graph:
        st.header("Pupil/teacher x NOX regression")
        g = sns.jointplot(data=df, x='NOX', y='INDUS', kind="reg", truncate=False)
        st.pyplot(g)
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
