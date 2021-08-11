import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import *

sns.set_style("darkgrid")

option = st.sidebar.radio(
    'Choose dataset',
    ['boston', 'iris', 'diabetes']
)

if option == 'boston':
    st.title("Boston housing dataset")

    graph_menu = st.sidebar.multiselect('Choose graphs', [
        "CRIM Histogram",
        "Charles River pricing effect",
        "Age x Industrialization",
        "Pupil/teacher x NOX regression"
        ])

    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['target'] = pd.Series(boston.target)

    st.header("Raw data")
    st.write(df)

    if "CRIM Histogram" in graph_menu:
        st.header("CRIM Histogram")
        g = sns.displot(df, x='CRIM')
        st.pyplot(g)

    if "Charles River pricing effect" in graph_menu:
        st.header("Charles River pricing effect")
        g = sns.catplot(data=df, x='CHAS', y='target', kind = 'swarm')
        st.pyplot(g)

    if "Age x Industrialization" in graph_menu:
        st.header("Age x Industrialization")
        g = sns.relplot(data=df, x='AGE', y='INDUS')
        st.pyplot(g)

    if "Pupil/teacher x NOX regression" in graph_menu:
        st.header("Pupil/teacher x NOX regression")
        g = sns.jointplot(data=df, x='NOX', y='INDUS', kind="reg", truncate=False)
        st.pyplot(g)

elif option == 'iris':
    st.title("Iris flower dataset")

    graph_menu = st.sidebar.multiselect('Choose graphs', [
            "Boxplots"
        ])

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = pd.Series(iris.target)

    st.header('Raw data')
    st.write(df)

    if "Boxplots" in graph_menu:
        st.header("Boxplots")
        b = st.selectbox("", df.columns)

        g = sns.catplot(data=df, y=b, kind='box')
        st.pyplot(g)

elif option == 'diabetes':
    st.title("Diabetes dataset")

    graph_menu = st.sidebar.multiselect('Choose graphs', [
            "Boxplots"
        ])

    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = pd.Series(diabetes.target)

    st.header('Raw data')
    st.write(df)

    if "Boxplots" in graph_menu:
        st.header("Boxplots")
        b = st.selectbox("", df.columns)

        g = sns.catplot(data=df, y=b, kind='box')
        st.pyplot(g)

else:
    pass
