# Importing necessary Modules
import numpy as np
import pandas as pd
import streamlit as st
import os
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#Loading some  of  the saved functions,model from joblib


model = joblib.load('house_model.pkl')
encoder = joblib.load('ordinary_encoder_house_model.pkl')
scaler = joblib.load('scaler_house_model.pkl')
X_train = joblib.load('X_train.pkl')
original = joblib.load('original1_data.csv')


#Encoding the cities and property with same saved encoder for efficiency
cities = encoder.categories_[0]
property_types = encoder.categories_[1]

#Making the app interactive by accepting user inputs
City_options = list(cities)
property_type_options = list(property_types)
city = st.selectbox('city', City_options)
PropertyType = st.selectbox('Property Type', property_type_options)

NbrLivingUnits = st.number_input(
    'Number of separate livable homes inside the property',
    min_value= 1, max_value=5, value= 1,
    help='''1= single home ,2= duplex, 3= Triplex,
            4to 5 = Multi unit  building ''')

SqFtLot = st.number_input(
    'Total land area of the property in square feet',
    min_value= 500, max_value=1000000, value= 5000,
    help = 'on average most  land area  cover  about 5,000 to 10,000 ')

SqFtTotLiving = st.number_input(
    'Total interior living space of the house in square feet',
    min_value= 300, max_value=11000, value= 1600,
    help = '''on average most living area cover up 
    to about  1500 to 2500 square feet area''')

SqFtFinBasement = st.number_input(
    'Total Area for the basement fo the building ',min_value= 0,
    max_value=3500, value= 0,
    help ='Enter 0 if u do not wish for the building to have basement')

Bathrooms = st.number_input(
    'Number of bathrooms', min_value= 1, max_value=10, value= 3)

Bedrooms = st.number_input(
    'Number of bedrooms',  min_value= 1, max_value=40, value= 3)

BldgGrade = st.number_input(
    'rate the building on scale of 1 to 13', min_value= 1,
    max_value=13, value= 5,
    help= '''Note this parameter is what affects thee
     prediction of the model the most hence be sure to evaluate ur input''')

YrBuilt = st.number_input(
    'The year the house was built from range of  1900 to 2018',
    min_value= 1900, max_value=2018, value= 2012 ,
    help= 'For best performance  of the model follow the limit given  ')

TrafficNoise = st.slider(
    '''Traffic Noise level(How close to the road
    
    0=no noise(not close to the road)5 = high noise hence (close to the road)''',
    min_value=0, max_value=5, step=1)

NewConstruction = st.checkbox('Is the house a new construction/renovated recently',
                                help= 'Likely is the house renovated recently')


#Transformedall the user input into one dataframe   for prediction
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


#Conditiion for the prediction and if yes hence prediction is carried out based on the user input
if st.button('Predict house price'):
    prediction = model.predict(input_data)
    st.success(f'Predicted price is ${prediction[0]:,.2f}')
df = pd.read_csv('C:/Users/CHIDOZIE/Downloads/Projeect  dataset/house_sales.csv')


#Defined a function that will recommend houses to the user still based on their input
# of  which was also used for prediction
def recommend_house(input_data, X_train, original1, top_n=5):
    similarity = cosine_similarity(input_data, X_train).flatten()
    top_idx = np.argsort(similarity)[-top_n:][::-1]
    if st.button('recommend houses'):
        st.subheader('Similar houses you may be interested in ')


#Combine the initial values for each columns using only the recommended ones and tracking
        # their real values from the original df saved to joblib
        recommended_houses = original1.iloc [top_idx][[
           'city',
           'PropertyType',
           'YrBuilt',
           'BldgGrade',
           'SqFtLot',
           'SqFtTotLiving',
           'SqFtFinBasement',
            'SalePrice',
       ]].copy()

#Prompted that recommendation should be based on the house with highest similarity scores
        recommended_houses['similarity_scores%'] = (similarity[top_idx]).round(2)

        #Renamed the colums of the recommended houses for better understanding of the user
        recommended_houses.rename(columns={'city': 'City',
    'PropertyType': 'Property Type',
    'YrBuilt': 'Year Built',
    'BldgGrade': 'Building Grade',
    'SqFtLot': 'Total land Size',
    'SqFtTotLiving': 'Living Area',
    'SqFtFinBasement': 'Basement Area'
}, inplace=True)

        st.dataframe(recommended_houses.reset_index(drop=True)) #Put  the recommended houses into dataframe
        csv = recommended_houses.to_csv(index=False).encode('utf-8') ##here converted the dataframe of the
        ##recommended df to csv and encoded it
        st.download_button("Download Recommendations", data=csv, file_name="recommendations.csv")#Downlaod button
recommend_house(input_data, X_train, original, top_n=5)#Called the defined function with the user input


#For clarification  added some sidebar notes to  the   user so they can understand some of
# the uncommon features of the house features or required inputs
st.sidebar.title('About   the APP')

st.sidebar.write('''
    This is a house predictive model  that accepts user input as the 
    features and predicts the price based on the features provided
    Hints:
    ** It is adviced to input values very realistic as the determines
    the perrformance of the model 
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

st.subheader("XGBoost Feature Importance – WEIGHT")


#Agraph of feature   importance displayed to the user  as  the predict or change values
# showing which features matters most
import shap
import matplotlib.pyplot as plt
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_data)
fig, ax = plt.subplots(figsize=(10,6))
shap.summary_plot(shap_values, input_data, plot_type="bar", show=False,color= 'red')
st.pyplot(fig)
plt.close(fig)