import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival.")

with st.form("prediction_form"):
    passenger_id = st.number_input("Passenger ID", min_value=1, value=999)
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
    sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, value=0)
    parch = st.number_input("Parents / Children Aboard", min_value=0, value=0)
    fare = st.number_input("Fare", min_value=0.0, value=7.25)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

    submit_button = st.form_submit_button("Predict")

if submit_button:
    sex_map = {"male": 0, "female": 1}
    embarked_map = {"S": 0, "C": 1, "Q": 2}

    input_data = pd.DataFrame([{
        "PassengerId": passenger_id,
        "Pclass": pclass,
        "Sex": sex_map[sex],
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_map[embarked]
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Survived")
    else:
        st.error(f"Prediction: Did Not Survive")

    st.write(f"Probability of survival: {probability:.2%}")