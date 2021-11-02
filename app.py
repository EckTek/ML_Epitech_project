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
    df_temp = df_temp.drop(columns=['newDevelopment', 'district', 'id'])
    # df_temp["status"] = df_temp['status'].replace('good', '0').astype(float)
    # df_temp["status"] = df_temp['status'].replace('renew', '1').astype(float)
    col1,col2, col3 = st.columns([3,3,3])
    buttonPress = False

    with col1:
#        Input["gender"] = st.radio('Tenant gender', ['male', 'female', 'mixed'])
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
        Input['price'] = 0
        features = ['bathroomsNumber', 'hasExterior', 'hasLift', 'propertyType', 'roomsNumber', 'size', 'status', 'hasParking']

#        df1 = pd.DataFrame(df_temp, columns=df_columns)
        df = pd.DataFrame.from_records([(Input)])
        df_temp.append(df)

        # print("TMP", df_temp.shape)
        class_features = ['propertyType']
        target = ['price']
        # print(df.head)
#        print("DF: ", df.shape)
        train, test = train_test_split(df_temp, test_size=0.2)
        encoder = preprocessing.LabelEncoder()

        # for class_feature in class_features:
        #     train[class_feature] = encoder.fit_transform(train[class_feature])
        #     test[class_feature] = encoder.fit_transform(test[class_feature])

        print("SHAPE: ", train.shape)
        # print("X SHAPE: ", df.shape)
        # print("Y SHAPE: ", df_temp.shape)
        model = LinearRegression()
        model.fit(train[features], train[target])
        model.fit(df.values, df_temp)

        # toto = model.predict(test[features])
        # print("TOTO: ", toto.shape)
        # fig, ax = plt.subplots(figsize=(30, 10))
        # sns.regplot(x='predictions', y='price', data=test)


        # conditions = np.where(
        #       (df_temp['bathroomsNumber'] == Input['bathroomsNumber'])
        #     & (df_temp['propertyType'] == Input['propertyType'])
        #     & (df_temp['roomsNumber'] == Input['roomsNumber'])
        #     & (df_temp['hasExterior'] == Input['hasExterior'])
        #     & (df_temp['hasParking'] == Input['hasParking'])
        #     & (df_temp['hasLift'] == Input['hasExterior'])
        #     & (df_temp['status'] == Input['status'])
        #     & (df_temp['size'] == Input['size']))
        
        # df = df_temp.loc[conditions]
        col3, col4, col5 = st.columns([2, 2, 2])
        with col3:
#            df = remove_outliers(df)
            st.write(df_temp)
#            st.write("Legend of my plot")
#            random_plot()
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