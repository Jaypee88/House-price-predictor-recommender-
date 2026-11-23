import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# App configuration and visual theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Washington Housing Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#14243B"
NAVY_LIGHT = "#243B59"
GOLD = "#C5A15A"
TEAL = "#5B8E8A"
WHITE = "#FFFFFF"
CHART_COLORS = [GOLD, NAVY_LIGHT, TEAL, "#9AABBC", "#B96B62"]

st.markdown(
    """
    <style>
      .stApp { background: #F5F3ED; color: #14243B; }
      [data-testid="stHeader"] { background: rgba(245,243,237,0.92); }
      [data-testid="stSidebar"] { background: #14243B; }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #F5F3ED;
      }
      h1, h2, h3 { color: #14243B; letter-spacing: -0.02em; }
      div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E8E3D8;
        border-left: 3px solid #C5A15A; border-radius: 10px;
        padding: 7px 10px; min-height: 76px;
      }
      div[data-testid="stMetricLabel"] {
        color: #687588; font-size: 0.76rem; line-height: 1.05;
      }
      div[data-testid="stMetricValue"] {
        color: #14243B; font-size: 1.22rem; line-height: 1.15;
      }
      .stButton > button, .stDownloadButton > button {
        background: #C5A15A; color: #14243B; border: 1px solid #C5A15A;
        border-radius: 9px; font-weight: 650; min-height: 2.65rem;
      }
      .stButton > button:hover, .stDownloadButton > button:hover {
        background: #D5B979; color: #14243B; border-color: #D5B979;
      }
      [data-testid="stAlert"] { border-radius: 10px; }
      .hero {
        background: linear-gradient(112deg, #14243B 0%, #243B59 100%);
        border: 1px solid #2A405D; border-radius: 16px;
        padding: 16px 22px; margin: 4px 0 12px 0;
      }
      .hero-kicker {
        color: #E8D8B5; font-size: 0.74rem; font-weight: 700;
        letter-spacing: 0.16em; text-transform: uppercase;
      }
      .hero-title {
        color: #FFFFFF; font-size: 1.65rem; font-weight: 750;
        margin: 5px 0 4px 0; line-height: 1.2;
      }
      .hero-copy { color: #D6DFE9; font-size: 0.95rem; margin: 0; }
      .section-note { color: #687588; font-size: 0.9rem; margin-top: -8px; }
      .property-price { color: #14243B; font-size: 1.35rem;
                        font-weight: 750; margin: 4px 0; }
      .property-meta { color: #687588; font-size: 0.86rem; }
      div[data-testid="stDataFrame"] { border: 1px solid #E8E3D8;
                                      border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Load model artefacts and the human-readable dataset
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts(base_dir):
    model = joblib.load(os.path.join(base_dir, "house_model.pkl"))
    encoder = joblib.load(os.path.join(base_dir, "ordinary_encoder_house_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler_house_model.pkl"))
    X_train = joblib.load(os.path.join(base_dir, "X_train.pkl"))
    return model, encoder, scaler, X_train


@st.cache_data
def load_original_data(base_dir):
    return pd.read_csv(os.path.join(base_dir, "original_data.csv"))


model, encoder, scaler, X_train = load_artifacts(BASE_DIR)
original = load_original_data(BASE_DIR)

if "SalePrice" not in original.columns:
    st.error("The original dataset must contain a SalePrice column.")
    st.stop()

# Preserve source row order: X_train.index refers to these original records.
# This also allows recommendations to display unencoded source values.
original = original.copy()
original.index = pd.RangeIndex(start=0, stop=len(original), step=1)

cities = list(encoder.categories_[0])
property_types = list(encoder.categories_[1])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def money(value):
    return f"USD {value:,.0f}"


def chart_style(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=WHITE,
        font=dict(color=NAVY, family="Arial"),
        margin=dict(l=12, r=12, t=52, b=12),
        legend_title_text="",
        title_font=dict(size=15, color=NAVY),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#ECE8DF", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ECE8DF", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def calculate_holdout_evaluation(
    original_df,
    train_index_values,
    feature_columns,
    _encoder,
    _scaler,
    _model,
):
    """
    Rebuild the notebook's 20% hold-out partition.

    X_train.pkl retains source dataframe indices from the shuffled split.
    Rows not present in those indices are the test records. They are
    transformed with the saved fitted encoder and scaler before evaluation.
    """
    train_indices = pd.Index(train_index_values)
    unknown_indices = train_indices.difference(original_df.index)
    if len(unknown_indices):
        raise ValueError("Saved training indices do not match original_data.csv.")

    test_indices = original_df.index.difference(train_indices)
    if len(test_indices) == 0:
        raise ValueError("No held-out records could be identified.")

    features = original_df.drop(
        columns=["SalePrice", "ZipCode", "state"],
        errors="ignore",
    ).copy()

    missing_features = [col for col in feature_columns if col not in features.columns]
    if missing_features:
        raise ValueError(
            "The original dataset is missing model features: "
            + ", ".join(missing_features)
        )

    categorical_columns = ["city", "PropertyType"]
    features[categorical_columns] = _encoder.transform(features[categorical_columns])
    features["NewConstruction"] = features["NewConstruction"].astype(int)

    scaled_columns = ["SqFtLot", "SqFtTotLiving", "SqFtFinBasement"]
    features[scaled_columns] = _scaler.transform(features[scaled_columns])
    features = features.reindex(columns=feature_columns)

    X_test = features.loc[test_indices]
    y_test = original_df.loc[test_indices, "SalePrice"].astype(float)
    predictions = _model.predict(X_test)

    results = pd.DataFrame(
        {
            "Actual Price": y_test.to_numpy(),
            "Predicted Price": predictions,
        },
        index=test_indices,
    )
    results["Residual"] = results["Actual Price"] - results["Predicted Price"]
    results["Absolute Error"] = results["Residual"].abs()

    metrics = {
        "R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "MSE": mean_squared_error(y_test, predictions),
        "Median AE": median_absolute_error(y_test, predictions),
        "Test Rows": len(y_test),
    }
    return results.reset_index(drop=True), metrics


def render_hero(kicker, title, copy):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">{kicker}</div>
          <div class="hero-title">{title}</div>
          <p class="hero-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# Sidebar navigation and project information
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏡 Housing Intelligence")
    st.markdown(
        "<span style='color:#E8D8B5;font-size:0.76rem;letter-spacing:0.12em;'>"
        "WASHINGTON STATE · USA</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "EXPLORE THE APP",
        ["Price Prediction", "Market Analytics", "Recommendations"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### About this project")
    st.caption(
        "An XGBoost regression model estimates residential sale prices. "
        "A separate similarity-based engine retrieves comparable properties "
        "from the original, human-readable housing records."
    )
    st.markdown("---")
    st.caption(
        "Research project using historical Washington State sales data. "
        "Predictions are estimates, not formal property valuations."
    )


# ---------------------------------------------------------------------
# PAGE 1 — PRICE PREDICTION
# ---------------------------------------------------------------------
if page == "Price Prediction":
    render_hero(
        "PROPERTY ESTIMATOR",
        "Estimate a property's market value",
        "Enter the property's characteristics to generate an estimated sale price "
        "using the trained XGBoost regression model.",
    )

    st.subheader("Property details")
    st.markdown(
        '<p class="section-note">Use the property’s known details. '
        "The model transforms categorical and numerical inputs with the saved "
        "preprocessing objects.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        city = st.selectbox("City", cities)
        property_type = st.selectbox("Property type", property_types)
        living_units = st.number_input(
            "Separate living units", min_value=1, max_value=5, value=1,
            help="For example, a single home, duplex, triplex or multi-unit building.",
        )
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=40, value=3)
        bathrooms = st.number_input(
            "Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.25
        )

    with col2:
        lot_size = st.number_input(
            "Land area (sq ft)", min_value=500, max_value=1_000_000, value=5_000,
            step=100,
        )
        living_area = st.number_input(
            "Interior living area (sq ft)", min_value=300, max_value=11_000,
            value=1_600, step=50,
        )
        basement_area = st.number_input(
            "Finished basement area (sq ft)", min_value=0, max_value=3_500,
            value=0, step=50,
        )

    with col3:
        building_grade = st.number_input(
            "Building grade", min_value=1, max_value=13, value=7,
            help="A quality grade on the dataset's 1–13 scale.",
        )
        year_built = st.number_input(
            "Year built", min_value=1900, max_value=2018, value=2012,
        )
        traffic_noise = st.slider("Traffic noise level", min_value=0, max_value=5, value=0)
        new_construction = st.checkbox("New construction / recently renovated")

    if st.button("Estimate house price", type="primary", use_container_width=True):
        user_input = pd.DataFrame(
            {
                "PropertyType": [property_type],
                "NbrLivingUnits": [living_units],
                "SqFtLot": [lot_size],
                "SqFtTotLiving": [living_area],
                "SqFtFinBasement": [basement_area],
                "Bathrooms": [bathrooms],
                "Bedrooms": [bedrooms],
                "BldgGrade": [building_grade],
                "YrBuilt": [year_built],
                "TrafficNoise": [traffic_noise],
                "NewConstruction": [int(new_construction)],
                "city": [city],
            }
        )

        encoded_columns = ["city", "PropertyType"]
        scaled_columns = ["SqFtLot", "SqFtTotLiving", "SqFtFinBasement"]
        user_input[encoded_columns] = encoder.transform(user_input[encoded_columns])
        user_input[scaled_columns] = scaler.transform(user_input[scaled_columns])
        user_input = user_input.reindex(columns=X_train.columns)

        estimated_price = float(model.predict(user_input)[0])
        st.session_state["input_data"] = user_input
        st.session_state["predicted_price"] = estimated_price
        st.session_state["property_details"] = {
            "City": city,
            "Property Type": property_type,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Living Area (SqFt)": living_area,
            "Land Area (SqFt)": lot_size,
            "Building Grade": building_grade,
        }

    if "predicted_price" in st.session_state:
        st.markdown("---")
        left, right = st.columns([1.25, 1])
        with left:
            st.markdown("### Estimated sale price")
            st.markdown(
                f"""
                <div style="background:#14243B;border-radius:14px;padding:22px 24px;">
                  <div style="color:#E8D8B5;font-size:0.78rem;letter-spacing:0.12em;">
                    MODEL ESTIMATE
                  </div>
                  <div style="color:#FFFFFF;font-size:2.15rem;font-weight:750;margin-top:5px;">
                    {money(st.session_state["predicted_price"])}
                  </div>
                  <div style="color:#D6DFE9;font-size:0.86rem;margin-top:8px;">
                    Based on the property characteristics entered above
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("### Property summary")
            details = st.session_state.get("property_details", {})
            st.write(f"**Location:** {details.get('City', '—')}")
            st.write(f"**Property type:** {details.get('Property Type', '—')}")
            st.write(
                f"**Bedrooms / bathrooms:** {details.get('Bedrooms', '—')} / "
                f"{details.get('Bathrooms', '—')}"
            )
            st.write(
                f"**Living area:** {details.get('Living Area (SqFt)', 0):,} sq ft"
            )
            st.caption(
                "This estimate is based on historical training data and should "
                "not be treated as a guaranteed selling price."
            )


# ---------------------------------------------------------------------
# PAGE 2 — MARKET AND MODEL ANALYTICS
# ---------------------------------------------------------------------
elif page == "Market Analytics":
    render_hero(
        "DATA SCIENCE WORKSPACE",
        "Washington housing market analytics",
        "A compact view of housing market patterns, property characteristics "
        "and predictive model performance.",
    )

    data = original.copy()
    data["city"] = data["city"].fillna("Unknown")
    total_rows = len(data)
    number_of_cities = data["city"].nunique()
    number_of_types = data["PropertyType"].nunique()
    median_price = data["SalePrice"].median()
    average_price = data["SalePrice"].mean()
    missing_cells = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    median_living_area = data["SqFtTotLiving"].median()
    new_build_share = data["NewConstruction"].astype(bool).mean() * 100

    try:
        evaluation, metrics = calculate_holdout_evaluation(
            original,
            X_train.index.tolist(),
            X_train.columns.tolist(),
            encoder,
            scaler,
            model,
        )
        evaluation_error = None
    except Exception as exc:
        evaluation = None
        metrics = None
        evaluation_error = str(exc)

    # Keep only the six highest-priority KPIs as cards. Additional descriptive
    # figures remain visible in a compact text strip to make more room for charts.
    st.subheader("Analytics at a glance")
    top_metrics = st.columns(3, gap="small")
    top_metrics[0].metric("Property records", f"{total_rows:,}")
    top_metrics[1].metric("Median sale price", money(median_price))
    top_metrics[2].metric(
        "Held-out R²", f"{metrics['R2']:.3f}" if metrics else "N/A"
    )

    model_metrics = st.columns(3, gap="small")
    model_metrics[0].metric(
        "Mean absolute error", money(metrics["MAE"]) if metrics else "N/A"
    )
    model_metrics[1].metric(
        "Root mean squared error", money(metrics["RMSE"]) if metrics else "N/A"
    )
    model_metrics[2].metric(
        "Test records", f"{metrics['Test Rows']:,}" if metrics else "N/A"
    )

    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:1px solid #E8E3D8;border-radius:10px;
                    padding:9px 13px;margin:2px 0 12px 0;color:#536176;
                    font-size:0.78rem;line-height:1.7;">
          <b style="color:#14243B;">Dataset profile</b>
          &nbsp; Average price: <b>{money(average_price)}</b>
          &nbsp; · &nbsp; Cities: <b>{number_of_cities:,}</b>
          &nbsp; · &nbsp; Property types: <b>{number_of_types:,}</b>
          &nbsp; · &nbsp; Median living area: <b>{median_living_area:,.0f} sq ft</b>
          &nbsp; · &nbsp; New builds: <b>{new_build_share:.1f}%</b>
          &nbsp; · &nbsp; Missing values: <b>{missing_cells:,}</b>
          &nbsp; · &nbsp; Duplicates: <b>{duplicate_rows:,}</b>
          &nbsp; · &nbsp; Median absolute error:
          <b>{money(metrics['Median AE']) if metrics else 'N/A'}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Model scores use reconstructed hold-out records. The original notebook "
        "fitted preprocessing before splitting, so the encoder and scaler had "
        "access to the full feature dataset; a future model refresh should fit "
        "preprocessing on training rows only."
    )
    if evaluation_error:
        st.caption(f"Model evaluation unavailable: {evaluation_error}")

    st.markdown("---")

    # Three compact visualisations per row. The smaller chart height and
    # reduced margins are intentional so several charts fit in one screenshot.
    st.subheader("1. Housing market patterns")
    row1 = st.columns(3, gap="small")

    price_hist = px.histogram(
        data,
        x="SalePrice",
        nbins=45,
        title="Sale price distribution",
        labels={"SalePrice": "Sale price (USD)", "count": "Properties"},
        color_discrete_sequence=[GOLD],
    )
    price_hist.update_xaxes(tickprefix="$", tickformat="~s")
    with row1[0]:
        st.plotly_chart(
            chart_style(price_hist, height=245),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    type_order = (
        data.groupby("PropertyType")["SalePrice"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    type_box = px.box(
        data,
        x="PropertyType",
        y="SalePrice",
        category_orders={"PropertyType": type_order},
        color="PropertyType",
        color_discrete_sequence=CHART_COLORS,
        title="Prices by property type",
        labels={"PropertyType": "Type", "SalePrice": "Sale price (USD)"},
    )
    type_box.update_yaxes(tickprefix="$", tickformat="~s")
    with row1[1]:
        st.plotly_chart(
            chart_style(type_box, height=245),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    city_volume = (
        data.groupby("city", dropna=False)
        .agg(Transactions=("SalePrice", "size"), MedianPrice=("SalePrice", "median"))
        .sort_values("Transactions", ascending=False)
        .head(8)
        .reset_index()
        .rename(columns={"city": "City"})
    )
    city_chart = px.bar(
        city_volume.sort_values("Transactions"),
        x="Transactions",
        y="City",
        orientation="h",
        title="Top cities by sales volume",
        color_discrete_sequence=[NAVY_LIGHT],
        hover_data={"MedianPrice": ":,.0f"},
        labels={"Transactions": "Sales"},
    )
    with row1[2]:
        st.plotly_chart(
            chart_style(city_chart, height=245),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    st.subheader("2. Location and property characteristics")
    row2 = st.columns(3, gap="small")

    city_medians = (
        data.groupby("city")
        .agg(Transactions=("SalePrice", "size"), MedianPrice=("SalePrice", "median"))
        .query("Transactions >= 25")
        .sort_values("MedianPrice", ascending=False)
        .head(8)
        .reset_index()
        .rename(columns={"city": "City", "MedianPrice": "Median price"})
    )
    city_price_chart = px.bar(
        city_medians.sort_values("Median price"),
        x="Median price",
        y="City",
        orientation="h",
        title="Median price by city",
        color_discrete_sequence=[GOLD],
        hover_data={"Transactions": True},
        labels={"Median price": "Median price (USD)"},
    )
    city_price_chart.update_xaxes(tickprefix="$", tickformat="~s")
    with row2[0]:
        st.plotly_chart(
            chart_style(city_price_chart, height=245),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    scatter_data = data.sample(n=min(1800, len(data)), random_state=42)
    living_scatter = px.scatter(
        scatter_data,
        x="SqFtTotLiving",
        y="SalePrice",
        color="PropertyType",
        color_discrete_sequence=CHART_COLORS,
        opacity=0.62,
        title="Living area vs sale price",
        labels={
            "SqFtTotLiving": "Living area (sq ft)",
            "SalePrice": "Sale price (USD)",
        },
        hover_data=["city", "Bedrooms", "Bathrooms", "BldgGrade"],
    )
    living_scatter.update_yaxes(tickprefix="$", tickformat="~s")
    with row2[1]:
        st.plotly_chart(
            chart_style(living_scatter, height=245),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    bedroom_data = data[data["Bedrooms"].between(1, 8)]
    bedroom_box = px.box(
        bedroom_data,
        x="Bedrooms",
        y="SalePrice",
        title="Prices by bedroom count",
        color_discrete_sequence=[TEAL],
        labels={"Bedrooms": "Bedrooms", "SalePrice": "Sale price (USD)"},
    )
    bedroom_box.update_yaxes(tickprefix="$", tickformat="~s")
    with row2[2]:
        st.plotly_chart(
            chart_style(bedroom_box, height=245),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    st.subheader("3. Feature relationships and model drivers")
    row3 = st.columns([2, 1], gap="small")
    correlation_columns = [
        "SalePrice",
        "SqFtTotLiving",
        "SqFtLot",
        "SqFtFinBasement",
        "Bedrooms",
        "Bathrooms",
        "BldgGrade",
        "YrBuilt",
    ]
    available_correlation_columns = [
        column for column in correlation_columns if column in data.columns
    ]
    correlation_matrix = data[available_correlation_columns].corr(numeric_only=True)
    heatmap = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale=["#F5F3ED", GOLD, NAVY],
        title="Correlation between property features",
        aspect="auto",
    )
    heatmap.update_layout(coloraxis_colorbar=dict(thickness=10, len=0.72))
    with row3[0]:
        st.plotly_chart(
            chart_style(heatmap, height=290),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    importance = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(8)
    importance_chart = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top model features",
        color_discrete_sequence=[GOLD],
        labels={"Importance": "Importance"},
    )
    with row3[1]:
        st.plotly_chart(
            chart_style(importance_chart, height=290),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )

    st.subheader("4. Model diagnostics")
    if evaluation is not None:
        row4 = st.columns(3, gap="small")

        axis_min = float(
            min(evaluation["Actual Price"].min(), evaluation["Predicted Price"].min())
        )
        axis_max = float(
            max(evaluation["Actual Price"].max(), evaluation["Predicted Price"].max())
        )
        actual_predicted = px.scatter(
            evaluation.sample(n=min(1800, len(evaluation)), random_state=42),
            x="Actual Price",
            y="Predicted Price",
            opacity=0.55,
            color_discrete_sequence=[GOLD],
            title="Actual vs predicted",
            labels={
                "Actual Price": "Actual (USD)",
                "Predicted Price": "Predicted (USD)",
            },
        )
        actual_predicted.add_shape(
            type="line",
            x0=axis_min,
            y0=axis_min,
            x1=axis_max,
            y1=axis_max,
            line=dict(color=NAVY, dash="dash", width=2),
        )
        actual_predicted.update_xaxes(tickprefix="$", tickformat="~s")
        actual_predicted.update_yaxes(tickprefix="$", tickformat="~s")
        with row4[0]:
            st.plotly_chart(
                chart_style(actual_predicted, height=245),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

        residual_hist = px.histogram(
            evaluation,
            x="Residual",
            nbins=42,
            title="Prediction error distribution",
            labels={"Residual": "Actual − predicted (USD)", "count": "Records"},
            color_discrete_sequence=[TEAL],
        )
        residual_hist.add_vline(x=0, line_dash="dash", line_color=NAVY)
        residual_hist.update_xaxes(tickprefix="$", tickformat="~s")
        with row4[1]:
            st.plotly_chart(
                chart_style(residual_hist, height=245),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

        residual_scatter = px.scatter(
            evaluation.sample(n=min(1800, len(evaluation)), random_state=42),
            x="Predicted Price",
            y="Residual",
            opacity=0.55,
            color_discrete_sequence=[NAVY_LIGHT],
            title="Error vs predicted price",
            labels={
                "Predicted Price": "Predicted (USD)",
                "Residual": "Actual − predicted (USD)",
            },
        )
        residual_scatter.add_hline(y=0, line_dash="dash", line_color=GOLD)
        residual_scatter.update_xaxes(tickprefix="$", tickformat="~s")
        residual_scatter.update_yaxes(tickprefix="$", tickformat="~s")
        with row4[2]:
            st.plotly_chart(
                chart_style(residual_scatter, height=245),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )

        with st.expander("Detailed error analysis and evaluation sample"):
            evaluation_segments = evaluation.copy()
            evaluation_segments["Actual price quartile"] = pd.qcut(
                evaluation_segments["Actual Price"],
                q=4,
                duplicates="drop",
            ).astype(str)
            segment_summary = (
                evaluation_segments.groupby("Actual price quartile", observed=False)
                .agg(
                    Records=("Absolute Error", "size"),
                    MAE=("Absolute Error", "mean"),
                    Mean_Error=("Residual", "mean"),
                )
                .reset_index()
                .rename(
                    columns={
                        "MAE": "Mean absolute error",
                        "Mean_Error": "Mean signed error",
                    }
                )
            )
            st.markdown("**Error by actual-price segment**")
            st.dataframe(
                segment_summary.style.format(
                    {
                        "Mean absolute error": "USD {:,.0f}",
                        "Mean signed error": "USD {:,.0f}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("**Sample of held-out predictions**")
            st.dataframe(
                evaluation[
                    ["Actual Price", "Predicted Price", "Residual", "Absolute Error"]
                ].head(100).style.format(
                    {
                        "Actual Price": "USD {:,.0f}",
                        "Predicted Price": "USD {:,.0f}",
                        "Residual": "USD {:,.0f}",
                        "Absolute Error": "USD {:,.0f}",
                    }
                ),
                use_container_width=True,
            )
    else:
        st.warning(
            "The market and feature visualisations are available, but model "
            "diagnostics could not be reconstructed from the saved files."
        )

    st.caption(
        "Interpretation note: correlations describe associations in this dataset; "
        "they do not establish that one feature causes a change in house price. "
        "Model feature importance is also not a causal explanation."
    )


# PAGE 3 — PROPERTY RECOMMENDATIONS
# ---------------------------------------------------------------------
else:
    render_hero(
        "SIMILAR PROPERTY FINDER",
        "Explore houses similar to your estimate",
        "Retrieve comparable records using feature similarity and closeness "
        "to the estimated sale price. Recommendations are displayed from the "
        "original dataset so property details remain readable.",
    )

    if "input_data" not in st.session_state or "predicted_price" not in st.session_state:
        st.info("Start on the Price Prediction page and estimate a property first.")
    else:
        input_data = st.session_state["input_data"].reindex(columns=X_train.columns)
        predicted_price = float(st.session_state["predicted_price"])

        left, right = st.columns([1, 2])
        with left:
            st.metric("Your estimated price", money(predicted_price))
        with right:
            st.markdown(
                "The match score combines similarity across the model features "
                "with similarity in historical sale price. It is a ranking score, "
                "not a probability that a property is available."
            )

        if st.button("Find similar properties", type="primary", use_container_width=True):
            # Align original records to X_train's exact row order.
            # Similarity positions refer to X_train positions, not CSV positions.
            train_original = original.loc[X_train.index].copy()
            train_prices = train_original["SalePrice"].to_numpy(dtype=float)

            feature_similarity = cosine_similarity(input_data, X_train).ravel()
            feature_range = feature_similarity.max() - feature_similarity.min()
            if feature_range > 0:
                feature_similarity = (
                    feature_similarity - feature_similarity.min()
                ) / feature_range

            price_difference = np.abs(train_prices - predicted_price)
            price_similarity = 1 - (
                price_difference / (price_difference.max() + 1e-9)
            )

            final_score = (0.70 * feature_similarity) + (0.30 * price_similarity)
            top_positions = np.argsort(final_score)[::-1][:5]

            selected = train_original.iloc[top_positions].copy()
            selected["Match Score (%)"] = (
                final_score[top_positions] * 100
            ).round(1)

            display_columns = [
                "city",
                "PropertyType",
                "NbrLivingUnits",
                "SqFtLot",
                "SqFtTotLiving",
                "SqFtFinBasement",
                "Bathrooms",
                "Bedrooms",
                "BldgGrade",
                "YrBuilt",
                "TrafficNoise",
                "NewConstruction",
                "SalePrice",
                "Match Score (%)",
            ]
            recommended = selected[display_columns].copy()
            recommended = recommended.rename(
                columns={
                    "city": "City",
                    "PropertyType": "Property Type",
                    "NbrLivingUnits": "Living Units",
                    "SqFtLot": "Land Size (SqFt)",
                    "SqFtTotLiving": "Living Area (SqFt)",
                    "SqFtFinBasement": "Basement Area (SqFt)",
                    "BldgGrade": "Building Grade",
                    "YrBuilt": "Year Built",
                    "TrafficNoise": "Traffic Noise Level",
                    "NewConstruction": "New / Renovated",
                    "SalePrice": "Price (USD)",
                }
            )
            recommended["New / Renovated"] = recommended["New / Renovated"].map(
                {True: "Yes", False: "No", 1: "Yes", 0: "No"}
            )

            st.session_state["recommendations"] = recommended.reset_index(drop=True)

        if "recommendations" in st.session_state:
            recommended = st.session_state["recommendations"]
            st.markdown("---")
            st.subheader("Your closest matches")
            st.caption(
                "Prices, locations and property attributes below come from "
                "the original, unencoded property records."
            )

            card_columns = st.columns(3)
            for position, (_, house) in enumerate(recommended.iterrows()):
                with card_columns[position % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{house['City']} · {house['Property Type']}**")
                        st.markdown(
                            f'<div class="property-price">{money(house["Price (USD)"])}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="property-meta">'
                            f'{int(house["Bedrooms"])} beds · {house["Bathrooms"]} baths'
                            f'<br>{int(house["Living Area (SqFt)"]):,} sq ft living area'
                            f'<br>{int(house["Land Size (SqFt)"]):,} sq ft land'
                            f'<br>Built {int(house["Year Built"])}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"**Match score:** {house['Match Score (%)']:.1f}%"
                        )

            st.markdown("#### Compare recommended properties")
            st.dataframe(recommended, hide_index=True, use_container_width=True)

            csv_data = recommended.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download recommendations (CSV)",
                data=csv_data,
                file_name="washington_house_recommendations.csv",
                mime="text/csv",
                use_container_width=True,
            )
