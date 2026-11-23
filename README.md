# Washington Housing Intelligence
## House Price Prediction & Recommendation System

An interactive Streamlit application that combines an XGBoost regression model with a similarity-based property recommender. It uses historical residential sales data from Washington State, USA.

## Application features

### 1. House price prediction
- Accepts property details such as city, property type, bedrooms, bathrooms, living area, land size, building grade and year built.
- Applies the saved ordinal encoder and scaler used during model training.
- Returns an estimated sale price from the trained XGBoost regressor.

### 2. Market analytics dashboard
The analytics page explores the original, human-readable housing data through:
- Dataset coverage, property counts, city and property-type counts, price averages and medians, missing values and duplicate rows.
- Sale-price distribution and sale-price comparisons by property type.
- Transaction volume and median sale prices across cities.
- Living area versus sale price and sale-price patterns across bedroom counts.
- Correlation analysis for selected numerical property features.
- Model evaluation on the held-out test records: R², MAE, RMSE and median absolute error.
- Actual-versus-predicted, residual-distribution and residual-versus-predicted charts.
- Error analysis by actual-price quartile and model-reported feature importance.

The held-out evaluation is reconstructed using the original row indices retained in `X_train.pkl`. The saved encoder and scaler are applied to the corresponding records before evaluation, so the displayed scores are calculated against records not used to fit the model.

### 3. Similar-property recommendations
- Calculates feature similarity using cosine similarity.
- Combines feature similarity (70%) and closeness to the predicted sale price (30%).
- Aligns each encoded training row with its corresponding original dataset record.
- Displays actual property values, sale prices and match scores.
- Supports CSV export.

Recommendations are similar historical records from the dataset, not confirmation that a property is currently listed or available.

## Technology stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly
- Joblib

## Project files

```text
House-price-predictor-recommender-/
├── House price app.py
├── House predictive model.ipynb
├── house_model.pkl
├── ordinary_encoder_house_model.pkl
├── scaler_house_model.pkl
├── X_train.pkl
├── original_data.csv
├── house_sales.csv
├── requirements.txt
└── README.md
```

## Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Jaypee88/House-price-predictor-recommender-.git
   cd House-price-predictor-recommender-
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit application:

   ```bash
   streamlit run "House price app.py"
   ```

## Dataset and limitations

The project uses historical housing sales data from Washington State. The model is trained for this dataset and geographic scope; predictions should not be assumed to generalise to other states or countries. They are estimates for learning and demonstration purposes, not formal appraisals or guaranteed sale prices.

## Live application

[Open the Streamlit app](https://washingtonhousemodel.streamlit.app/)

## Author

Nwankwo Chidozie Johnpaul
