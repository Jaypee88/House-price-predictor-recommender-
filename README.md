.

🏠 Smart House Price Prediction & Recommendation System
📌 Overview

This project is an AI-powered House Price Prediction and Recommendation System that helps users:

Predict the price of a house based on input features
Discover similar houses within a close price range and feature similarity

The system combines machine learning prediction with a similarity-based recommendation engine to provide a more complete and practical real estate experience.

🚀 Features  
💰 House Price Prediction using machine learning   \
🏠 Smart House Recommendations based on:   
Predicted price  
Feature similarity  
📊 Interactive web app built with Streamlit  
📥 Downloadable recommendation results  
🎯 User-friendly interface with guided inputs  
🛠️ Technologies Used  
Python   
Streamlit  
Pandas  
NumPy  
Scikit-learn  
XGBoost  
Matplotlib  

📂 Project Structure
house-recommendation-system/
│
├── house_app.py                  # Streamlit application  
├── house_model.pkl              # Trained XGBoost model  
├── ordinary_encoder_house_model.pkl  
├── scaler_house_model.pkl  
├── X_train.pkl
├── original_data.csv            # Cleaned dataset for recommendations  
├── requirements.txt  
└── README.md  

⚙️ How It Works  
🔹 1. Data Processing  
Original dataset contains house sales data from Washington, USA
Additional feature engineering was performed:
Extracted city information from ZIP codes using geolocation tools
Cleaned and prepared dataset for training
🔹 2. Price Prediction  
Model: XGBoost Regressor  
Inputs include:  
City  
Property type  
Bedrooms, bathrooms  
Living area, land size  
Building grade  
Year built  
Traffic noise  
Renovation status  

👉 The model predicts the estimated house price

🔹 3. House Recommendation

After prediction:

The system compares user input with existing houses
Uses:  
Cosine similarity (features)
Price similarity (closeness to predicted price)

👉 Final score combines both to recommend the most relevant houses

▶️ How to Run the Project  
Clone the repository  
git clone https://github.com/your-username/house-recommendation-system.git   
cd house-recommendation-system  
Install dependencies  
pip install -r requirements.txt  
Run the app  
streamlit run house_app.py  
🌐 Live Demo  
(Add your Streamlit link here after deployment)  

📊 Dataset  
Based on real estate data from Washington, USA
Includes features such as:  
Property type  
Location (city derived from ZIP code)  
Size and structure details  
Sale price  

⚠️ Limitations  
Model is trained only on data from Washington State
Predictions may not generalize well to other regions
Accuracy depends on realistic user inputs  
📈 Future Improvements  
Expand dataset to multiple states/countries  
Improve model accuracy with more data  
Add map-based visualization  
Integrate real-time housing data APIs  
Add user accounts and saved recommendations  


👤 Author  

Nwankwo Chidozie Johnpaul

📜 License

This project is for educational and research purposes.

To Check or experience the app follow the link below
https://washingtonhousemodel.streamlit.app/
