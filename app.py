import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# ======================================
# CONFIG
# ======================================

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Disease Risk Stratification",
    page_icon="🏥",
    layout="wide"
)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Prediction Dashboard",
        "Project Overview",
        "Dataset Information",
        "ML Pipeline",
        "Model Performance",
        "About Developer"
    ]
)

# ======================================
# PAGE 1
# ======================================

if page == "Prediction Dashboard":

    st.title("🏥 Disease Risk Stratification")

    st.markdown(
        """
        Predict Chronic Kidney Disease (CKD) risk
        using a stacking ensemble machine learning model.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        sg = st.number_input(
            "Specific Gravity",
            value=1.020,
            format="%.3f"
        )

        al = st.slider(
            "Albumin",
            0,
            5,
            1
        )

        su = st.slider(
            "Sugar",
            0,
            5,
            0
        )

        sc = st.number_input(
            "Serum Creatinine",
            value=1.2
        )

        hemo = st.number_input(
            "Hemoglobin",
            value=15.0
        )

    with col2:

        pcv = st.number_input(
            "Packed Cell Volume",
            value=44.0
        )

        rbcc = st.number_input(
            "RBC Count",
            value=5.2
        )

        htn = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

        dm = st.selectbox(
            "Diabetes Mellitus",
            ["No", "Yes"]
        )

        appet = st.selectbox(
            "Appetite",
            ["Good", "Poor"]
        )

    if st.button(
        "Predict Risk",
        use_container_width=True
    ):

        payload = {

            "sg": sg,
            "al": al,
            "su": su,
            "sc": sc,
            "hemo": hemo,
            "pcv": pcv,
            "rbcc": rbcc,

            "htn":
                1 if htn == "Yes" else 0,

            "dm":
                1 if dm == "Yes" else 0,

            "appet":
                1 if appet == "Poor" else 0
        }

        try:

            response = requests.post(
                API_URL,
                json=payload
            )

            result = response.json()

            prediction = result["prediction"]

            probability = result["probability"]

            st.divider()

            colA, colB = st.columns(2)

            with colA:

                st.metric(
                    "Risk Probability",
                    f"{probability*100:.2f}%"
                )

            with colB:

                if prediction == 1:

                    st.error(
                        "🔴 High Risk (CKD)"
                    )

                else:

                    st.success(
                        "🟢 Low Risk"
                    )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",

                    value=probability * 100,

                    title={
                        "text":
                        "Disease Risk Score"
                    },

                    gauge={
                        "axis":
                            {"range":[0,100]},

                        "steps":[

                            {
                                "range":[0,40],
                                "color":"lightgreen"
                            },

                            {
                                "range":[40,70],
                                "color":"gold"
                            },

                            {
                                "range":[70,100],
                                "color":"salmon"
                            }
                        ]
                    }
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.subheader(
                "Clinical Recommendation"
            )

            if prediction == 1:

                st.warning(
                    """
                    Elevated CKD risk detected.

                    Consider:
                    - Kidney function tests
                    - Serum creatinine monitoring
                    - Nephrologist consultation
                    """
                )

            else:

                st.success(
                    """
                    Risk appears low.

                    Continue:
                    - Routine health checks
                    - Healthy lifestyle
                    - Periodic monitoring
                    """
                )

        except Exception as e:

            st.error(
                f"API Error: {e}"
            )

# ======================================
# PAGE 2
# ======================================

elif page == "Project Overview":

    st.title("📌 Project Overview")

    st.markdown(
        """
        ## Disease Risk Stratification using Ensemble Machine Learning

        Healthcare systems often need to identify
        high-risk patients early.

        This project uses an ensemble learning
        approach to classify Chronic Kidney Disease
        risk using clinical patient records.

        ### Objectives

        - Predict CKD risk
        - Reduce missed diagnoses
        - Assist clinicians
        - Demonstrate end-to-end ML deployment

        ### Ensemble Models

        - Logistic Regression
        - Random Forest
        - XGBoost

        Final prediction is generated using
        Stacked Generalization.
        """
    )

# ======================================
# PAGE 3
# ======================================

elif page == "Dataset Information":

    st.title("📊 Dataset Information")

    st.markdown(
        """
        ### Dataset

        Chronic Kidney Disease Dataset

        Total Records:
        - 400 Patients

        Target Variable:
        - CKD
        - Not CKD

        Features:
        - Age
        - Blood Pressure
        - Albumin
        - Sugar
        - Serum Creatinine
        - Hemoglobin
        - Packed Cell Volume
        - RBC Count
        - Hypertension
        - Diabetes Mellitus
        - Appetite
        and more...

        Missing values were handled
        using statistical imputation.
        """
    )

# ======================================
# PAGE 4
# ======================================

elif page == "ML Pipeline":

    st.title("⚙️ Machine Learning Pipeline")

    st.markdown(
        """
        ### Data Preprocessing

        ✔ Missing Value Imputation

        ✔ Label Encoding

        ✔ Feature Scaling

        ### Class Balancing

        ✔ SMOTE

        ### Feature Selection

        ✔ Mutual Information

        ✔ Recursive Feature Elimination

        ### Models

        ✔ Logistic Regression

        ✔ Random Forest

        ✔ XGBoost

        ### Ensemble

        ✔ Stacking Classifier

        ### Deployment

        ✔ FastAPI

        ✔ Streamlit Dashboard
        """
    )

# ======================================
# PAGE 5
# ======================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    performance = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Stacking Ensemble"
        ],

        "F1 Score":[
            0.99,
            1.00,
            1.00,
            1.00
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.bar_chart(
        performance.set_index("Model")
    )

# ======================================
# PAGE 6
# ======================================

elif page == "About Developer":

    st.title("👨‍💻 About Developer")

    st.markdown(
        """
        ### Developer Information

        **Name:** Sayambar Roy Chowdhury

        **GitHub:**  
        https://github.com/Sayambar2004

        **LinkedIn:**  
        https://www.linkedin.com/in/sayambar-roy-chowdhury-731b0a282/

        **Email:**  
        sayambarroychowdhury@gmail.com

        ---
        Developed as an end-to-end Machine Learning
        healthcare project using:

        - Scikit-Learn
        - XGBoost
        - Imbalanced-Learn
        - FastAPI
        - Streamlit
        """
    )