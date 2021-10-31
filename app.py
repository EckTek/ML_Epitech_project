import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def remove_outliers(df, category = None, input = None):
    if category is None or input is None:
        Q1 = df['math score'].quantile(0.25)
        Q3 = df['math score'].quantile(0.75)
    else:
        Q1 = df[df[category] == input[category]]['math score'].quantile(0.25)
        Q3 = df[df[category] == input[category]]['math score'].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df['math score'] < (Q1 - 1.5 * IQR)) | (df['math score'] > (Q3 + 1.5 * IQR)))]
    return df

def random_plot():
    x = np.random.exponential(2, 10000)
    st.line_chart(data=x, width=0, height=0)

def main():
    # title
    html_title = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Price housing calculation in Barcelona <i class="fas fa-home"></i></h1>
    </div>
    """
    st.markdown(html_title, unsafe_allow_html=True)

    df = pd.read_csv('ml_dataset.csv')
    Input = {}

    st.markdown("Welcome to our website ! The purpose of this website is to calculate the average price of your future accomodation, according to your selected criteria. Have fun !")

    col1,col2 = st.columns([2,2])
    buttonPress = False
    with col1:
        Input["gender"] = st.radio('Tenant gender', ['male', 'female', 'mixed'])

    with col2:
        Input["bathrooms"] = st.selectbox('Bathroom', ["1", "2", "3"])

        buttonPress = st.button('Predict')
        if buttonPress:
            print(Input)

    if buttonPress:
        col3, col4, col5 = st.columns([2, 2, 2])
        with col3:
#            df = remove_outliers(df)
            st.write("Legend of my plot")
            random_plot()
        # CODE EXAMPLE FOR A COLUMN
        # with col4:
        #     st.write("Here you can see the median of math score for female people")
        #     df = remove_outliers(df, "gender", Input)
        #     fig = px.box(df[df["gender"] == Input["gender"]], y="math score", x="gender", width=450, height=450)
        #     st.plotly_chart(fig)
        with col4:
            st.write("Legend of my plot")
            random_plot()
        with col5:
            st.write("Legend of my plot")
            random_plot()
#       IF YOU WANT MORE COLUMN...
#         col6, col7, col8 = st.columns([2, 2, 2])


if __name__ == '__main__':
	main()