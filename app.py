
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Streamlit - Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "best_used_car_price_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
    model_error = None

except Exception as error:
    model_loaded = False
    model_error = str(error)


# --------------------------------------------------
# GET CAR NAMES FROM TRAINED MODEL
# --------------------------------------------------

def get_car_names(trained_model):
    try:
        preprocessing = trained_model.named_steps["preprocessing"]

        categorical = preprocessing.named_transformers_["categorical"]

        encoder = categorical.named_steps["onehot"]

        car_names = encoder.categories_[0].tolist()

        return sorted(car_names)

    except Exception:
        return [
            "ciaz",
            "city",
            "corolla altis",
            "fortuner",
            "innova",
            "swift",
            "verna"
        ]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🚗 AutoPredict")

    st.caption("Used Car Price Prediction System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "Predict Price", "About"]
    )

    st.divider()

    st.caption("Machine Learning Project")

    st.caption("Powered by Gradient Boosting")


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.title("🚘 Used Car Price Prediction")

    st.subheader(
        "Predict the estimated resale value of a used vehicle."
    )

    st.write(
        "This application uses machine learning techniques "
        "to estimate used car selling prices based on vehicle "
        "characteristics and historical data."
    )

    st.divider()

    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Selected Model",
            value="Gradient Boosting"
        )

    with col2:
        st.metric(
            label="Training Records",
            value="239"
        )

    with col3:
        st.metric(
            label="Testing Records",
            value="60"
        )

    st.divider()

    st.subheader("How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 1. Enter Details")
        st.write(
            "Provide the vehicle's specifications, "
            "including its age, mileage and other details."
        )

    with step2:
        st.markdown("### 2. Run Prediction")
        st.write(
            "The trained machine-learning model "
            "processes the information."
        )

    with step3:
        st.markdown("### 3. View Estimate")
        st.write(
            "The application displays the estimated "
            "resale price of the vehicle."
        )

    st.info(
        "Go to 'Predict Price' in the sidebar to estimate "
        "the selling price of a vehicle."
    )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predict Price":

    st.title("💰 Used Car Price Prediction")

    st.write(
        "Enter the vehicle details below to generate "
        "an estimated selling price."
    )

    st.divider()

    if not model_loaded:

        st.error("The trained model could not be loaded.")

        st.code(model_error)

        st.warning(
            "Check that best_used_car_price_model.joblib "
            "is in the same folder as app.py."
        )

    else:

        car_names = get_car_names(model)

        st.subheader("Vehicle Information")

        with st.form("car_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:

                car_name = st.selectbox(
                    "Car Name",
                    options=car_names
                )

                present_price = st.number_input(
                    "Present Price (₹ Lakh)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=5.0,
                    step=0.1
                )

                kms_driven = st.number_input(
                    "Kilometres Driven",
                    min_value=0,
                    max_value=1000000,
                    value=50000,
                    step=1000
                )

                fuel_type = st.selectbox(
                    "Fuel Type",
                    ["Petrol", "Diesel", "CNG"]
                )

            with col2:

                seller_type = st.selectbox(
                    "Seller Type",
                    ["Dealer", "Individual", "Trustmark Dealer"]
                )

                transmission = st.selectbox(
                    "Transmission",
                    ["Manual", "Automatic"]
                )

                owner = st.selectbox(
                    "Previous Owners",
                    [0, 1, 2, 3]
                )

                vehicle_year = st.number_input(
                    "Manufacturing Year",
                    min_value=2000,
                    max_value=2026,
                    value=2018,
                    step=1
                )

            st.divider()

            submitted = st.form_submit_button(
                "🔍 Predict Selling Price",
                use_container_width=True
            )

        # ------------------------------------------
        # GENERATE PREDICTION
        # ------------------------------------------

        if submitted:

            vehicle_age = 2026 - int(vehicle_year)

            input_data = pd.DataFrame([{
                "Car_Name": car_name,
                "Present_Price": float(present_price),
                "Kms_Driven": int(kms_driven),
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Owner": int(owner),
                "Vehicle_Age": int(vehicle_age)
            }])

            try:

                prediction = float(model.predict(input_data)[0])

                rupees_value = prediction * 100000

                st.divider()

                st.subheader("Prediction Result")

                st.success(
                    "Your vehicle price prediction was generated successfully!"
                )

                st.markdown("### Estimated Selling Price")

                price_col1, price_col2 = st.columns(2)

                with price_col1:

                    st.metric(
                        label="Predicted Price",
                        value=f"₹{prediction:,.2f} Lakh"
                    )

                with price_col2:

                    st.metric(
                        label="Equivalent Amount",
                        value=f"₹{rupees_value:,.0f}"
                    )

                st.divider()

                st.subheader("Vehicle Summary")

                summary = pd.DataFrame({
                    "Vehicle Detail": [
                        "Car Name",
                        "Present Price",
                        "Kilometres Driven",
                        "Fuel Type",
                        "Seller Type",
                        "Transmission",
                        "Previous Owners",
                        "Vehicle Age"
                    ],

                    "Value": [
                        car_name,
                        f"₹{present_price:,.2f} Lakh",
                        f"{kms_driven:,} km",
                        fuel_type,
                        seller_type,
                        transmission,
                        owner,
                        f"{vehicle_age} years"
                    ]
                })

                st.dataframe(
                    summary,
                    use_container_width=True,
                    hide_index=True
                )

                st.info(
                    "This prediction is an estimate generated by "
                    "your trained machine-learning model. "
                    "Actual selling prices may differ depending "
                    "on vehicle condition, location and market demand."
                )

                st.caption(
                    "Currency: Indian Rupees (INR). "
                    "The original dataset expresses prices in lakhs."
                )

            except Exception as error:

                st.error("An error occurred while generating the prediction.")

                st.code(str(error))


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.title("ℹ️ About This Project")

    st.subheader(
        "Predictive Modeling of Used Car Prices "
        "Using Multiple Regression and Ensemble Regressors"
    )

    st.write(
        """
        This project applies machine learning techniques to
        estimate the selling prices of used vehicles.

        Several regression algorithms were developed and
        evaluated, including:

        - Multiple Linear Regression
        - Decision Tree Regression
        - Random Forest Regression
        - Gradient Boosting Regression

        The models were evaluated using:

        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R-squared (R²)

        Gradient Boosting was selected as the saved prediction
        model because it achieved the lowest Mean Absolute Error
        among the evaluated models.
        """
    )

    st.divider()

    st.subheader("Technology Used")

    st.write(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- Joblib\n"
        "- Streamlit"
    )

    st.divider()

    st.caption(
        "Academic Machine Learning Project"
    )
