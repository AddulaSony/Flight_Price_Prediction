import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("Flight Price Prediction")

file = st.file_uploader("Upload Data_Train.xlsx", type=["xlsx"])

if file is not None:

    df = pd.read_excel(file)
    df.dropna(inplace=True)

    df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'])
    df['Journey_Day'] = df['Date_of_Journey'].dt.day
    df['Journey_Month'] = df['Date_of_Journey'].dt.month
    df.drop('Date_of_Journey', axis=1, inplace=True)

    df['Dep_hour'] = df['Dep_Time'].str.split(':').str[0].astype(int)
    df.drop('Dep_Time', axis=1, inplace=True)

    df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]
    df['Arrival_hour'] = df['Arrival_Time'].str.split(':').str[0].astype(int)
    df.drop('Arrival_Time', axis=1, inplace=True)

    df['Duration_hour'] = df['Duration'].str.extract('(\d+)').astype(int)
    df.drop('Duration', axis=1, inplace=True)

    df['Total_Stops'] = df['Total_Stops'].replace({
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4
    })

    df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    st.success("Model Trained Successfully")

    user_input = []
    for col in X.columns:
        value = st.number_input(f"{col}", value=0)
        user_input.append(value)

    if st.button("Predict"):
        prediction = model.predict([user_input])
        st.write("Estimated Price:", int(prediction[0]))
