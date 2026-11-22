
import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AutoPredict | AI Vehicle Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

REFERENCE_YEAR = 2026
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_optimized_used_car_price_model.joblib"


# ==========================================================
# 2. DESIGN
# ==========================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Oswald:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp, button, input, select, textarea {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #0d2a5c 0%, transparent 55%),
        linear-gradient(160deg, #050a14 0%, #071428 60%, #0a1f44 100%);
}
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 1320px;
    padding-top: 3.2rem;
    padding-bottom: 2rem;
}
p, li { color: #A9B8CC; }

.eyebrow {
    color: #60A5FA; font-weight: 500; font-size: 0.85rem;
    letter-spacing: 0.28em; margin-bottom: 1.1rem;
}
.hl {
    font-family: 'Oswald', 'Impact', sans-serif !important;
    font-weight: 700; font-size: 4rem; line-height: 1.06;
    color: #FFFFFF; letter-spacing: -0.005em; margin: 0 0 1.6rem 0;
}
.hl .blue { color: #4C93FF; display: block; }
.lead { color: #A9B8CC; font-size: 1.05rem; line-height: 1.7; max-width: 440px; margin-bottom: 2.2rem; }

.feature { display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1.4rem; }
.f-icon {
    flex: 0 0 52px; height: 52px; border-radius: 12px;
    background: #101a2c; border: 1px solid #25344d;
    display: flex; align-items: center; justify-content: center; color: #E6EDF7;
}
.f-icon svg { width: 26px; height: 26px; }
.f-title { color: #FFFFFF; font-weight: 700; font-size: 1.05rem; margin-bottom: 0.15rem; }
.f-text  { color: #8FA3B8; font-size: 0.93rem; line-height: 1.5; }

.cta {
    display: inline-flex; align-items: center; gap: 0.6rem; margin-top: 1rem;
    background: linear-gradient(180deg, #4C93FF, #2F6FE4);
    color: #FFFFFF; font-weight: 600; font-size: 1.05rem;
    padding: 0.85rem 1.6rem; border-radius: 14px;
    box-shadow: 0 10px 30px rgba(47,111,228,0.35);
}
.cta svg { width: 22px; height: 22px; }

[data-testid="stColumn"]:has(.window-marker),
[data-testid="column"]:has(.window-marker) {
    background: #0b1220;
    border: 1px solid #2a3a55;
    border-radius: 22px;
    padding: 1.1rem 1.3rem 1.6rem 1.3rem !important;
    box-shadow: 0 30px 80px rgba(0,0,0,0.55);
}
.win-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem; }
.dots { display: flex; gap: 0.5rem; }
.dot { width: 13px; height: 13px; border-radius: 50%; display: inline-block; }
.d-r { background: #FF5F57; } .d-y { background: #FEBC2E; } .d-g { background: #28C840; }
.pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    font-size: 0.82rem; font-weight: 600; padding: 0.4rem 0.9rem; border-radius: 8px;
}
.pill-on  { background: #1E6EF0; color: #FFFFFF; }
.pill-off { background: #4a1c1c; color: #FF9A9A; }
.pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; display: inline-block; }

[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid #1c2a42; }
[data-baseweb="tab"] { background: transparent !important; padding: 0.6rem 0.2rem !important; }
[data-baseweb="tab"] p { color: #A9B8CC; font-weight: 500; font-size: 0.95rem; }
[data-baseweb="tab"][aria-selected="true"] p { color: #60A5FA; font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: #3B82F6 !important; }
[data-baseweb="tab-border"] { display: none; }

.banner {
    position: relative; height: 330px; border-radius: 14px; overflow: hidden;
    margin: 0.8rem 0 1.2rem 0; border: 1px solid #1c2a42;
}
.banner svg.car { position: absolute; left: 50%; top: 46%; width: 78%; transform: translate(-50%, -50%); }

.ov-card {
    position: absolute; top: 20px; right: 22px;
    background: rgba(6,12,24,0.62); border: 1.5px dashed #4C93FF;
    padding: 0.9rem 1.2rem; text-align: center; min-width: 190px;
}
.ov-card .h { position: absolute; width: 11px; height: 11px; background: #FFFFFF; border: 2px solid #4C93FF; border-radius: 50%; }
.h.tl { top: -7px; left: -7px; } .h.tr { top: -7px; right: -7px; }
.h.bl { bottom: -7px; left: -7px; } .h.br { bottom: -7px; right: -7px; }
.ov-k { color: #B9C7D6; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; }
.ov-v { font-family: 'Oswald', sans-serif !important; color: #FFFFFF; font-weight: 700; font-size: 1.7rem; line-height: 1.2; }

.ov-bar {
    position: absolute; left: 0; right: 0; bottom: 0;
    background: rgba(5,9,16,0.82); padding: 0.9rem 1.4rem;
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.ov-side { flex: 1; }
.ov-small { color: #8FA3B8; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; }
.ov-big {
    font-family: 'Oswald', sans-serif !important; color: #FFFFFF;
    font-weight: 700; font-size: 1.75rem; line-height: 1.15;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ov-big.muted { color: #6B7C93; font-size: 1.3rem; font-weight: 600; }
.ov-div { width: 1px; align-self: stretch; background: #3a4a63; }

[data-testid="stForm"] { background: transparent; border: none; padding: 0; }
[data-testid="stWidgetLabel"] p { color: #A9B8CC !important; font-weight: 600 !important; font-size: 0.82rem !important; }
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background: #0f1b2e !important; border-radius: 8px !important; border-color: #26385a !important;
}
input { color: #E6EDF7 !important; }
[data-baseweb="select"] > div { border: 1px solid #26385a !important; }

[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(180deg, #4C93FF, #2F6FE4); color: #FFFFFF;
    border: none; border-radius: 10px; padding: 0.8rem 1rem;
    font-weight: 700; font-size: 1rem; margin-top: 0.4rem;
    box-shadow: 0 8px 24px rgba(47,111,228,0.3);
}
[data-testid="stFormSubmitButton"] button:hover { background: linear-gradient(180deg, #6AA5FF, #3C7DF0); color: #FFFFFF; }
[data-testid="stFormSubmitButton"] button:focus-visible { outline: 3px solid #9CC4FF; }

.stButton button {
    background: transparent; border: 1px solid #2f4468; color: #C9D6E6;
    border-radius: 10px; font-weight: 600; padding: 0.6rem 1rem;
}
.stButton button:hover { border-color: #4C93FF; color: #FFFFFF; background: #0f1b2e; }

.empty-note {
    color: #8FA3B8; background: #0f1b2e; border: 1px dashed #26385a;
    border-radius: 10px; padding: 1.2rem; text-align: center; margin-top: 1rem;
}
.footer { text-align: center; color: #5F7088; font-size: 0.8rem; margin-top: 2.4rem; }

@media (max-width: 900px) {
    .hl { font-size: 3rem; }
    .banner { height: 280px; }
    .ov-big { font-size: 1.3rem; }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ==========================================================
# 3. DESIGN HELPERS
# ==========================================================

def indian_format(number):
    """Format a number using Indian digit grouping."""

    number = int(round(number))
    sign = "-" if number < 0 else ""
    digits = str(abs(number))

    if len(digits) <= 3:
        return sign + digits

    last_three = digits[-3:]
    rest = digits[:-3]
    groups = []

    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]

    if rest:
        groups.insert(0, rest)

    return sign + ",".join(groups) + "," + last_three


def find_hero_image():

    for name in ["hero.jpg", "hero.jpeg", "hero.png", "hero.webp"]:

        image_path = BASE_DIR / name

        if image_path.exists():

            mime = (
                "image/png" if name.endswith("png")
                else "image/webp" if name.endswith("webp")
                else "image/jpeg"
            )

            encoded = base64.b64encode(
                image_path.read_bytes()
            ).decode()

            return f"data:{mime};base64,{encoded}"

    return None


HERO_IMAGE = find_hero_image()

CAR_SVG = (
    '<svg class="car" viewBox="0 0 600 220" xmlns="http://www.w3.org/2000/svg">'
    '<defs>'
    '<linearGradient id="body" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#e3e8ee"/><stop offset="1" stop-color="#7d8999"/>'
    '</linearGradient>'
    '<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">'
    '<stop offset="0" stop-color="#4C93FF" stop-opacity="0.35"/>'
    '<stop offset="1" stop-color="#4C93FF" stop-opacity="0"/>'
    '</radialGradient>'
    '</defs>'
    '<ellipse cx="300" cy="190" rx="280" ry="26" fill="url(#glow)"/>'
    '<ellipse cx="300" cy="188" rx="250" ry="9" fill="#000" opacity="0.5"/>'
    '<path d="M40 148 C40 128 62 118 102 114 L172 88 C204 66 252 56 312 56 '
    'C372 56 422 66 452 88 L512 110 C552 116 566 128 566 148 L566 158 L40 158 Z" fill="url(#body)"/>'
    '<path d="M188 92 C214 74 252 66 300 66 L300 106 L176 106 Z" fill="#16233a"/>'
    '<path d="M312 66 C360 66 398 74 424 94 L440 106 L312 106 Z" fill="#16233a"/>'
    '<rect x="296" y="66" width="6" height="42" fill="#9aa5b4"/>'
    '<rect x="518" y="124" width="36" height="9" rx="4" fill="#bfe0ff"/>'
    '<rect x="48" y="126" width="26" height="8" rx="4" fill="#ff5a5a" opacity="0.85"/>'
    '<circle cx="142" cy="158" r="36" fill="#0a0e15" stroke="#2a3444" stroke-width="4"/>'
    '<circle cx="142" cy="158" r="19" fill="#8b95a3"/><circle cx="142" cy="158" r="7" fill="#2a3444"/>'
    '<circle cx="474" cy="158" r="36" fill="#0a0e15" stroke="#2a3444" stroke-width="4"/>'
    '<circle cx="474" cy="158" r="19" fill="#8b95a3"/><circle cx="474" cy="158" r="7" fill="#2a3444"/>'
    '</svg>'
)

ICON_FORM = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 6h8M18 6h2M4 12h2M12 12h8M4 18h10M20 18h0"/>'
    '<circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/></svg>'
)

ICON_CHART = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M5 20V11M11 20V4M17 20v-6M3 20h18"/></svg>'
)

ICON_CHECK = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/></svg>'
)


def feature(icon, title, text):
    return (
        f'<div class="feature"><div class="f-icon">{icon}</div>'
        f'<div><div class="f-title">{title}</div><div class="f-text">{text}</div></div></div>'
    )


def banner_html(result):

    if HERO_IMAGE:
        background = (
            "linear-gradient(180deg, rgba(5,10,20,0.15), rgba(5,10,20,0.55)), "
            f"url({HERO_IMAGE}) center / cover no-repeat"
        )
        car = ""
    else:
        background = (
            "radial-gradient(ellipse at 70% 20%, #2a4468 0%, transparent 55%), "
            "linear-gradient(180deg, #101b2e 0%, #0a1322 60%, #070d18 100%)"
        )
        car = CAR_SVG

    overlay = ""

    if result is not None:

        overlay = (
            '<div class="ov-card">'
            '<i class="h tl"></i><i class="h tr"></i><i class="h bl"></i><i class="h br"></i>'
            '<div class="ov-k">ESTIMATED VALUE</div>'
            f'<div class="ov-v">₹{indian_format(result["prediction"])}</div>'
            '</div>'
        )

        left = (
            '<div class="ov-small">YOUR VEHICLE</div>'
            f'<div class="ov-big">{str(result["car_name"]).upper()}</div>'
        )

        right = (
            '<div class="ov-small">PREDICTED PRICE (INR)</div>'
            f'<div class="ov-big">₹ {indian_format(result["prediction"])}</div>'
        )

    else:

        left = '<div class="ov-small">YOUR VEHICLE</div><div class="ov-big muted">Choose a model below</div>'
        right = '<div class="ov-small">PREDICTED PRICE</div><div class="ov-big muted">Awaiting details</div>'

    return (
        f'<div class="banner" style="background: {background};">'
        f'{car}{overlay}'
        f'<div class="ov-bar"><div class="ov-side">{left}</div>'
        f'<div class="ov-div"></div><div class="ov-side">{right}</div></div>'
        '</div>'
    )


# ==========================================================
# 4. LOAD TRAINED MODEL
# ==========================================================

def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    model_loaded = True
    model_error = None

except Exception as error:
    model = None
    model_loaded = False
    model_error = str(error)


# ==========================================================
# 5. EXTRACT MODEL CATEGORIES
# ==========================================================

def get_model_categories(trained_model):
    """
    Extract categories from the fitted OneHotEncoder
    inside the trained model's ColumnTransformer.
    """

    if not hasattr(trained_model, "named_steps"):
        raise TypeError("The loaded model is not a Scikit-learn Pipeline.")

    if "preprocessor" not in trained_model.named_steps:
        raise KeyError("The model does not contain a 'preprocessor' step.")

    preprocessor = trained_model.named_steps["preprocessor"]

    if not hasattr(preprocessor, "named_transformers_"):
        raise RuntimeError("The model preprocessor has not been fitted.")

    encoder = preprocessor.named_transformers_["categorical"]

    if encoder is None or not hasattr(encoder, "categories_"):
        raise RuntimeError("The fitted categorical encoder was not found.")

    feature_names = [
        "Car_Name",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner"
    ]

    if len(feature_names) != len(encoder.categories_):
        raise RuntimeError("The encoder categories do not match the expected features.")

    return {
        feature: sorted(values.tolist())
        for feature, values in zip(feature_names, encoder.categories_)
    }


# ==========================================================
# 6. SESSION STATE
# ==========================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ==========================================================
# 7. PAGE LAYOUT
# ==========================================================

left_col, right_col = st.columns([1, 1.45], gap="large")


# ----------------------------------------------------------
# LEFT: INTRODUCTION
# ----------------------------------------------------------

with left_col:

    st.markdown(
        '<div class="eyebrow">ACCURATE. FAST. DATA-DRIVEN.</div>'
        '<div class="hl">Value Every Car.<span class="blue">Price It Right.</span></div>'
        '<div class="lead">Enter a used vehicle\'s details and get an estimated resale '
        'price in seconds, right on the page.</div>'
        + feature(ICON_FORM, "Enter the details", "Model, year, mileage, fuel type and ownership.")
        + feature(ICON_CHART, "Machine-learning estimate", "Optimized Random Forest trained on combined vehicle datasets.")
        + feature(ICON_CHECK, "A price you can read", "Displayed in Indian Rupees.")
        + f'<div class="cta">{ICON_CHECK}Instant estimate, no sign-up</div>',
        unsafe_allow_html=True
    )


# ----------------------------------------------------------
# RIGHT: APP WINDOW
# ----------------------------------------------------------

with right_col:

    status = (
        '<span class="pill pill-on"><i></i>Model online</span>'
        if model_loaded else
        '<span class="pill pill-off"><i></i>Model offline</span>'
    )

    st.markdown(
        '<div class="window-marker"></div>'
        '<div class="win-bar">'
        '<div class="dots"><span class="dot d-r"></span><span class="dot d-y"></span><span class="dot d-g"></span></div>'
        f'{status}'
        '</div>',
        unsafe_allow_html=True
    )

    tab_predict, tab_results, tab_about = st.tabs(
        ["Predict", "Results", "About"]
    )


    # ======================================================
    # TAB 1: PREDICT
    # ======================================================

    with tab_predict:

        banner_slot = st.empty()

        if not model_loaded:

            banner_slot.markdown(
                banner_html(None),
                unsafe_allow_html=True
            )

            st.error("The trained machine learning model could not be loaded.")
            st.code(model_error)

            st.warning(
                "Make sure best_optimized_used_car_price_model.joblib "
                "is located in the same directory as app.py."
            )

        else:

            try:
                categories = get_model_categories(model)

                car_names = categories["Car_Name"]
                fuel_options = categories["Fuel_Type"]
                seller_options = categories["Seller_Type"]
                transmission_options = categories["Transmission"]
                owner_options = categories["Owner"]

            except Exception as error:

                st.error("Unable to read the categories from the trained model.")
                st.exception(error)

                car_names = []
                fuel_options = []
                seller_options = []
                transmission_options = []
                owner_options = []

            if car_names and fuel_options and seller_options and transmission_options and owner_options:

                with st.form("vehicle_prediction_form"):

                    col1, col2 = st.columns(2, gap="large")

                    with col1:

                        car_name = st.selectbox(
                            "Vehicle Model",
                            options=car_names
                        )

                        kms_driven = st.number_input(
                            "Total Kilometres Driven",
                            min_value=0,
                            max_value=1000000,
                            value=50000,
                            step=1000
                        )

                        fuel_type = st.selectbox(
                            "Fuel Type",
                            options=fuel_options
                        )

                    with col2:

                        seller_type = st.selectbox(
                            "Seller Type",
                            options=seller_options
                        )

                        transmission = st.selectbox(
                            "Transmission",
                            options=transmission_options
                        )

                        # IMPORTANT:
                        # The trained model expects owner names as text,
                        # not numerical owner codes.
                        owner = st.selectbox(
                            "Previous Owners",
                            options=owner_options
                        )

                        vehicle_year = st.number_input(
                            "Manufacturing Year",
                            min_value=2000,
                            max_value=REFERENCE_YEAR,
                            value=2018,
                            step=1
                        )

                    submitted = st.form_submit_button(
                        "Estimate price",
                        use_container_width=True,
                        type="primary"
                    )

                # ==========================================
                # PROCESS PREDICTION
                # ==========================================

                if submitted:

                    vehicle_age = REFERENCE_YEAR - int(vehicle_year)

                    input_data = pd.DataFrame([{
                        "Car_Name": car_name,
                        "Kms_Driven": int(kms_driven),
                        "Fuel_Type": fuel_type,
                        "Seller_Type": seller_type,
                        "Transmission": transmission,
                        "Owner": owner,
                        "Vehicle_Age": int(vehicle_age)
                    }])

                    try:

                        prediction = float(model.predict(input_data)[0])

                        if not pd.notna(prediction):
                            raise ValueError("The model returned an invalid prediction.")

                        st.session_state.prediction_result = {
                            "prediction": prediction,
                            "car_name": car_name,
                            "kms_driven": int(kms_driven),
                            "fuel_type": fuel_type,
                            "seller_type": seller_type,
                            "transmission": transmission,
                            "owner": owner,
                            "vehicle_age": vehicle_age,
                            "vehicle_year": int(vehicle_year)
                        }

                    except Exception as error:

                        st.session_state.prediction_result = None

                        st.error("Unable to generate the prediction.")
                        st.exception(error)

                # ==========================================
                # DISPLAY PREDICTION RESULT
                # ==========================================

                result = st.session_state.prediction_result

                banner_slot.markdown(
                    banner_html(result),
                    unsafe_allow_html=True
                )

                if result is not None:

                    st.caption(
                        "This is an estimate from a trained model. Actual selling "
                        "prices vary with vehicle condition, location and market demand."
                    )

                    if st.button("Clear prediction", use_container_width=True):
                        st.session_state.prediction_result = None
                        st.rerun()

            else:
                st.error(
                    "The vehicle selection categories could not be loaded. "
                    "Check the model file and its preprocessing pipeline."
                )


    # ======================================================
    # TAB 2: RESULTS
    # ======================================================

    with tab_results:

        result = st.session_state.prediction_result

        if result is None:

            st.markdown(
                '<div class="empty-note">No estimate yet. Fill in the vehicle details '
                'on the Predict tab and select Estimate price.</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown("#### Vehicle summary")

            summary = pd.DataFrame({

                "Vehicle Specification": [
                    "Vehicle Model",
                    "Manufacturing Year",
                    "Kilometres Driven",
                    "Fuel Type",
                    "Seller Type",
                    "Transmission",
                    "Previous Owners",
                    "Vehicle Age"
                ],

                "Entered Value": [
                    result["car_name"],
                    str(result["vehicle_year"]),
                    f"{result['kms_driven']:,} km",
                    result["fuel_type"],
                    result["seller_type"],
                    result["transmission"],
                    result["owner"],
                    f"{result['vehicle_age']} years"
                ]

            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "Predicted Selling Price",
                f"₹{indian_format(result['prediction'])}"
            )


    # ======================================================
    # TAB 3: ABOUT
    # ======================================================

    with tab_about:

        st.markdown(
            "#### Predictive Modeling of Used Car Prices Using Multiple "
            "Regression and Ensemble Regressors"
        )

        st.write(
            "This project combines the Car Details from CarDekho dataset "
            "and the CarDekho vehicle dataset to develop a machine-learning "
            "system for predicting used vehicle selling prices."
        )

        st.write(
            "**Algorithms compared:** Multiple Linear Regression, "
            "Decision Tree, Random Forest and Gradient Boosting."
        )

        st.write(
            "**Evaluation metrics:** MAE, MSE, RMSE and R²."
        )

        st.write(
            "**Selected model:** Optimized Random Forest Regressor, "
            "selected based on its lowest Mean Absolute Error."
        )

        st.write(
            "**Dataset:** 3,875 vehicle records; 3,100 training records "
            "and 775 testing records."
        )

        st.write(
            "**Built with:** Python, Pandas, Scikit-learn, Joblib and Streamlit."
        )


# ==========================================================
# 8. FOOTER
# ==========================================================

st.markdown(
    '<div class="footer">AutoPredict | Academic Machine Learning Project</div>',
    unsafe_allow_html=True
)
