import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")

def price_prediction(inputs):
        # Create dataframes
        df_file = pd.read_csv('ml_dataset.csv')
        df_file = df_file.drop(columns=['newDevelopment', 'district', 'id'])

        df_input = pd.DataFrame.from_records([(inputs)])

        # Preparing properties for model
        features = ['bathroomsNumber', 'hasExterior', 'hasLift', 'propertyType', 'roomsNumber', 'size', 'status', 'hasParking']
        class_features = ['propertyType', 'status']
        target = ['price']

        # Convert string type columns to numeric type
        encoder = preprocessing.LabelEncoder()
        for class_feature in class_features:
            df_input[class_feature] = encoder.fit_transform(df_input[class_feature])
            df_file[class_feature] = encoder.fit_transform(df_file[class_feature])

        # Apply Linear Regression model
        model = LinearRegression()
        model.fit(df_file[features], df_file[target])
        price = model.predict(df_input)

        #Price is for one year so we divide here by 12
        final_price = price[0][0] / 12.0

        return round(final_price, 2)

def main():
    # title
    html_title = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Price housing calculation in Barcelona <i class="fas fa-home"></i></h1><br />
    </div>
    """
    st.markdown(html_title, unsafe_allow_html=True)
    st.markdown("""Welcome to our website ! The purpose of this website is to calculate the average price of your future accomodation, according to your selected criteria. Have fun !<br />""", unsafe_allow_html=True)
    
    # Create dict to store input values
    Input = {}

    col1,col2, col3 = st.columns([3,3,3])
    buttonPress = False

    with col1:
        Input["hasExterior"] = st.radio('Terrace', [0, 1])
        Input['status'] = st.selectbox('Status', ['good', 'renew'])

    with col2:
        Input["hasLift"] = st.radio('Lift', [0, 1])
        Input["bathroomsNumber"] = st.selectbox('Bathroom', [1, 2, 3, 4, 5])
        Input['propertyType'] = st.selectbox('Type of property', ['flat', 'studio', 'penthouse', 'duplex'])

    with col3:
        Input["hasParking"] = st.radio('Parking', [0, 1])
        Input['roomsNumber'] = st.selectbox("Number of rooms", [1, 2, 3, 4, 5, 6, 7])
        Input['size'] = st.selectbox('Size (in squared meter)', [x for x in range(22, 387)])

    buttonPress = st.button('Predict')

    if buttonPress:
        price = price_prediction(Input)

        col4, = st.columns(1)
        with col4:
            st.markdown(f"""
                <style>
                span.big-font {{
                    font-size:20px !important;
                    text-align: center;
                }}
                b {{ color: MEDIUMSEAGREEN }}
                </style>
                <span class="big-font">The average rent price will be <b>{price} €</b> / month</span>
                """, unsafe_allow_html=True)




if __name__ == '__main__':
	main()