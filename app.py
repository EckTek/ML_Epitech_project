import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

# hasExterior: 0 et 1
# hasLift : 0 et 1
# hasParking : 0 et 1
# bathroomsNumber ; 1-5
# Property type: [‘flat’, ‘studio’, ‘penthouse’, ‘duplex’]
# Status: [’good’, ‘renew’]
# RoomNumber = [1-7]
# Size: 22-387


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
    st.markdown("Welcome to our website ! The purpose of this website is to calculate the average price of your future accomodation, according to your selected criteria. Have fun !")

    # INIT dataset variables
    Input = {}
    df_temp = pd.read_csv('ml_dataset.csv')
    print("BEF: ", df_temp.columns)
    df_temp = df_temp.drop(columns=['newDevelopment', 'district', 'id'])
    print("COLUMNS: ", df_temp.columns)
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
        print("INPUT: ", Input)

    if buttonPress:
        features = ['bathroomsNumber', 'hasExterior', 'hasLift', 'propertyType', 'roomsNumber', 'size', 'status', 'hasParking']

        df = pd.DataFrame.from_records([(Input)])

        class_features = ['propertyType', 'status']
        target = ['price']
        encoder = preprocessing.LabelEncoder()

        for class_feature in class_features:
            df[class_feature] = encoder.fit_transform(df[class_feature])
            df_temp[class_feature] = encoder.fit_transform(df_temp[class_feature])

        model = LinearRegression()
        model.fit(df_temp[features], df_temp[target])

        toto = model.predict(df)
        col4,  = st.columns(1)
        with col4:
#            df = remove_outliers(df)
            # r_2 = model.score(df_temp[features], df_temp[target])
            # st.write(r_2)
            # st.write(df_temp)
            final_price = toto[0][0] / 12.00
            print(final_price)
            # html_cide = f"""
            #     <style>
            #     .big-font {
            #         font-size:100px !important;
            #     }
            #     </style>
            #     <p class="big-font">This is the {final_price}</p>
            #     """
            st.markdown(f"""
                <style>
                span.big-font {{
                    font-size:20px !important;
                    text-align: center;
                }}
                b {{ color: MEDIUMSEAGREEN }}
                </style>
                <span class="big-font">The average rent price will be <b>{final_price} €</b> / month</span>
                """, unsafe_allow_html=True)






if __name__ == '__main__':
	main()