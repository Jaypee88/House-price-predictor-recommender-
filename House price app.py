# Importing necessary Modules
import numpy as np
import pandas as pd
import streamlit as st
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizergit

#Loading some  of  the saved functions,model from joblib
model = joblib.load('house_model.pkl')
encoder = joblib.load('ordinary_encoder_house_model.pkl')
scaler = joblib.load('scaler_house_model.pkl')
X_train = joblib.load('X_train.pkl')
original = pd.read_csv('original_data.csv')


#Encoding the cities and property with same saved encoder for efficiency
cities = encoder.categories_[0]
property_types = encoder.categories_[1]

#Setting the UI of  the app
st.set_page_config(page_title="House Price Predictor", layout="wide")
st.title("🏠 House Price Prediction & Recommendation System")
st.markdown("A smart app to predict house prices and suggest similar properties.")


#Making the app interactive by accepting user inputs
City_options = list(cities)
property_type_options = list(property_types)

#Listing the two main function of the app in tabs
tab1, tab2 = st.tabs(['🏘️💵 House price prediction', '🏠✨ House Recommendation'])

# TAB1 for prediction of house prices.
with tab1:
    st.header("Enter House Details")
#Made a three column category to improve UI  in the app for all inputs
    col1, col2, col3 = st.columns(3)
#First column
    with col1:
        city = st.selectbox('City', City_options)

        PropertyType = st.selectbox('Property Type', property_type_options)

        NbrLivingUnits = st.number_input(
            'Number of separate livable homes inside the property', min_value=1, max_value=5,
            value=1,help='''1=single home ,2=duplex, 3= Triplex, 4to 5 = Multi unit  building ''')

        Bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=40, value=3)

        Bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=3)

    #Second column
    with col2:
        SqFtLot = st.number_input(
            'Total land area of the property in square feet',min_value= 500,max_value=1000000,
            value=5000,help='on average most land area cover about 5,000 to 10,000 ')

        SqFtTotLiving =st.number_input(
            'Total interior living space of the house in square feet',min_value= 300,
            max_value=11000, value= 1600,help = '''on average most living area cover 
            up to about  1500 to 2500 square feet area''')


        SqFtFinBasement = st.number_input(
            'Total Area for the basement fo the building',min_value=0,
        max_value=3500, value= 0,help ='Enter 0 if u do not wish for the building to have basement')

    #Third Column
    with col3:
        BldgGrade = st.number_input(
            'rate the building on scale of 1 to 13', min_value= 1,
        max_value=13, value= 5,help= '''Note this parameter is what affects thee
        prediction of the model the most hence be sure to evaluate ur input''')

        YrBuilt = st.number_input(
            'The year the house was built from range of  1900 to 2018',min_value= 1900,
            max_value=2018,value=2012,help='For optimal performance adhere to the limit provided')

        TrafficNoise = st.slider(
            '''Traffic Noise level(How close to the road 0=no noise(not close to the road)
            5 = high noise hence (close to the road)''',min_value=0, max_value=5, step=1)

        NewConstruction = st.checkbox(
            'Is the house a new construction/renovated recently',
            help= 'Likely if the house was renovated recently')

#Transformed user input to be received by model
    if st.button('🚀 Predict House Price'):
        input_data = pd.DataFrame({
            'PropertyType': [PropertyType],
            'NbrLivingUnits': [NbrLivingUnits],
            'SqFtLot': [SqFtLot],
            'SqFtTotLiving': [SqFtTotLiving],
            'SqFtFinBasement': [SqFtFinBasement],
            'Bathrooms': [Bathrooms],
            'Bedrooms': [Bedrooms],
            'BldgGrade': [BldgGrade],
            'YrBuilt': [YrBuilt],
            'TrafficNoise': [TrafficNoise],
            'NewConstruction': [(1 if NewConstruction else 0)],
            'city': [city]
        })
        #Encoded the required inputs from the user
        encoded_cols = ['city','PropertyType']

        #Applied scaling to the required user input for model optimization
        scaled_cols = ['SqFtLot','SqFtTotLiving','SqFtFinBasement']

        #Transformed the encoded and the scaled cols in the  user input dataframe
        input_data[encoded_cols] = encoder.transform(input_data[encoded_cols])
        input_data[scaled_cols] = scaler.transform(input_data[scaled_cols])

        # Save for recommendation
        st.session_state["input_data"] = input_data

        #For prediction
        prediction = model.predict(input_data)
        st.success(f"💰 Predicted Price: ${prediction[0]:,.2f}")

#  Features Importance visualization
if "input_data" in st.session_state:
    input_data = st.session_state["input_data"]

    st.subheader("Feature Importance")
    importance = model.feature_importances_
    features = input_data.columns

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    st.pyplot(fig)


# TAB 2: HOUSE RECOMMENDATION
with tab2:
    st.header("Recommend Similar houses")

    st.info("Uses your last input to recommend similar houses.")

    if st.button('🔍 Recommend Houses'):
        if "input_data" not in st.session_state:
            st.warning("⚠️ Please predict a house price first.")
        else:
            input_data = st.session_state["input_data"]

            # Ensure same column order
            input_data = input_data[X_train.columns]

            # ================= FEATURE SIMILARITY =================
            similarity = cosine_similarity(input_data, X_train).flatten()

            # Normalize
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-9)

            # ================= PRICE SIMILARITY =================
            predicted_price = model.predict(input_data)[0]

            # Use original dataset prices
            prices = original['SalePrice'].values[:len(X_train)]

            # Compute closeness to predicted price
            price_diff = np.abs(prices - predicted_price)

            price_similarity = 1 - (price_diff / (price_diff.max() + 1e-9))

            # ================= COMBINED SCORE =================
            final_score = (0.7 * similarity) + (0.3 * price_similarity)

            # Get top houses
            top_idx = np.argsort(final_score)[-5:][::-1]

            recommended_houses = original.iloc[top_idx][[
                'city',
                'PropertyType',
                'NbrLivingUnits',
                'SqFtLot',
                'SqFtTotLiving',
                'SqFtFinBasement',
                'Bathrooms',
                'Bedrooms',
                'BldgGrade',
                'YrBuilt',
                'TrafficNoise',
                'NewConstruction',
                'SalePrice'
            ]].copy()

            # Add scores
            recommended_houses['Match Score (%)'] = (final_score[top_idx] * 100).round(2)

            # Rename columns
            recommended_houses.rename(columns={
                'city': 'City',
                'PropertyType': 'Property Type',
                'NbrLivingUnits': 'Living Units',
                'SqFtLot': 'Land Size (SqFt)',
                'SqFtTotLiving': 'Living Area (SqFt)',
                'SqFtFinBasement': 'Basement Area',
                'Bathrooms': 'Bathrooms',
                'Bedrooms': 'Bedrooms',
                'BldgGrade': 'Building Grade',
                'YrBuilt': 'Year Built',
                'TrafficNoise': 'Traffic Noise Level',
                'NewConstruction': 'New/Renovated',
                'SalePrice': 'Price ($)'
            }, inplace=True)

            #Converted new construction from 0/1 to yes or now for better understanding
            recommended_houses['New/Renovated'] = recommended_houses[
                                                'New/Renovated'].map({1: 'Yes', 0: 'No'})

            st.success(f"💡 Showing houses close to predicted price: ${predicted_price:,.2f}")
            st.dataframe(recommended_houses.reset_index(drop=True))

            # Download button
            csv = recommended_houses.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Recommendations", data=csv, file_name="smart_recommendations.csv")

#For clarification  added some sidebar notes to  the   user so they can understand some of
# the uncommon features of the house features or required inputs
st.sidebar.title('About the APP')

st.sidebar.write('''
    This is a house predictive model  that accepts user input as the 
    features and predicts the price based on the features provided
    Hints:
    ** It is adviced to input values very realistic as the determines
    the perrformance of the model .
    Look down on the page for more clarity on the model parameters
''')

st.subheader('About Building grade')
st.write('''
    ** Building grade is the features that affects the house price thee 
    most. It  determines the level of the quality of the building
     like if furnished and also higher building grade implies better
     materials and infrastructures.
''')

st.subheader('About SqFtTotLiving ')
st.write('''
    ** The living Area is simply the expected land  space u want the 
    building to cover and how big the building is
    typical values lies in 600 to 1600 square feet.
''')

st.subheader('About SqFtLot ')
st.write('''
    ** Land Area is   how big thee compound should be as some building 
    can have same features but bigger land Area implies bigger price
      values typically lies in range of 500 to 15000 square feet
      on average but can accept up to 1000000land Area   
''')

st.warning('''House built ealier than the specified year of 2018 may be less
        accurate those within 2015  to 1900 are best for optimal performance
        ''')

st.write(original.columns.tolist())