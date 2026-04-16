import streamlit as st
import joblib
import pandas as pd

model=joblib.load("traffic_model.pkl")
feature=joblib.load("feature_columns.pkl")

st.title("Traffic Level Predictor 🚦")

st.write("""
This app uses a **Random Forest** model trained on the 
Metro Interstate Traffic Volume dataset from Kaggle.
It predicts traffic as Clear Road / Medium / Moving Slowly / Heavy Congestion based on 
time, day, month, weather, and temperature.
""")

hour = st.slider("Hour of Day", 0, 23, 12)

day_options = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
                }

day_label = st.selectbox("Day of Week", list(day_options.keys()))
day = day_options[day_label]


month_options = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
month_label = st.selectbox("Month", list(month_options.keys()))
month = month_options[month_label]

temp = st.number_input("Temperature (Kelvin)", value=288.0)
rain = st.number_input("Rain in last 1 hour (mm)", value=0.0)
snow = st.number_input("Snow in last 1 hour (mm)", value=0.0)
clouds = st.slider("Cloud Cover (%)", 0, 100, 40)

weather = st.selectbox("Weather Type", 
    ['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 
     'Haze', 'Fog', 'Thunderstorm', 'Snow', 'Squall', 'Smoke'])

holiday = st.selectbox("Holiday", 
    ['None', 'Columbus Day', 'Veterans Day', 'Thanksgiving Day',
     'Christmas Day', 'New Years Day', 'Washingtons Birthday',
     'Memorial Day', 'Independence Day', 'State Fair', 
     'Labor Day', 'Martin Luther King Jr Day'])


if st.button("Predict Traffic"):

    holiday_val=None if holiday=="None" else holiday

    input_data={"hour":hour,"day":day,"month":month,"temp":temp,"rain_1h":rain,
                "snow_1h":snow,"clouds_all":clouds,
                "weather_main":weather,"holiday":holiday_val
                }
    
    input_df=pd.DataFrame([input_data])
    input_df=pd.get_dummies(input_df,columns=["weather_main","holiday"])
    input_df=input_df.reindex(columns=feature, fill_value=0)

    prediction= model.predict(input_df)[0]


    st.success(f"🚦 Predicted Traffic Level: **{prediction}**")