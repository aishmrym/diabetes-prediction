import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}

[data-testid="metric-container"] {
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
BASE_DIR = Path(__file__).resolve().parent.parent

model = pickle.load(
    open(BASE_DIR / "model" / "diabetes_model.pkl", "rb")
)

scaler = pickle.load(
    open(BASE_DIR / "model" / "scaler.pkl", "rb")
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("ℹ️ Model Information")

st.sidebar.success("Random Forest Classifier")

st.sidebar.markdown("""
### Dataset
Pima Indians Diabetes Dataset

### Features
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### Model Performance
Accuracy: **86.36%**

### Team
Kelompok 6
""")

# =====================================
# HEADER
# =====================================
st.title("🩺 Diabetes Prediction Dashboard")

st.markdown("""
Predict the likelihood of diabetes using a Machine Learning model
trained on the Pima Indians Diabetes Dataset.
""")

st.divider()

# =====================================
# INPUT FORM
# =====================================
col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=1000,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

# =====================================
# PREDICTION
# =====================================
if st.button("🔍 Predict Diabetes Risk"):

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    probability = model.predict_proba(data_scaled)

    diabetes_prob = probability[0][1] * 100

    st.divider()

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            "Probability of Diabetes",
            f"{diabetes_prob:.2f}%"
        )

    with metric2:

        if diabetes_prob < 30:
            st.metric(
                "Risk Level",
                "🟢 Low"
            )

        elif diabetes_prob < 70:
            st.metric(
                "Risk Level",
                "🟡 Medium"
            )

        else:
            st.metric(
                "Risk Level",
                "🔴 High"
            )

    st.divider()

    if prediction[0] == 1:

        st.error(
            f"⚠️ High Risk of Diabetes ({diabetes_prob:.2f}%)"
        )

        st.warning("""
        The model predicts that the patient may have diabetes.

        This result is based on the input features provided and should
        not be considered a medical diagnosis.

        Please consult a healthcare professional for proper examination
        and diagnosis.
        """)

    else:

        st.success(
            f"✅ Low Risk of Diabetes ({diabetes_prob:.2f}%)"
        )

        st.info("""
        The model predicts that the patient is unlikely to have diabetes.

        Continue maintaining a healthy lifestyle, including balanced
        nutrition, regular physical activity, and routine health checkups.
        """)