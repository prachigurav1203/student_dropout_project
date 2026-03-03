import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("Models/xgb_final_model.pkl")
st.title("🎓 Student Dropout Prediction")

st.write("Enter Student Details (Numeric Encoded Values)")

# ---------- INPUTS ----------

School = st.number_input("School (0/1)", min_value=0)
Gender = st.number_input("Gender (0/1)", min_value=0)
Age = st.number_input("Age", min_value=10, max_value=25)
Address = st.number_input("Address (0/1)", min_value=0)
Family_Size = st.number_input("Family Size (0/1)", min_value=0)
Parental_Status = st.number_input("Parental Status (0/1)", min_value=0)

Mother_Education = st.number_input("Mother Education (0-4)", min_value=0, max_value=4)
Father_Education = st.number_input("Father Education (0-4)", min_value=0, max_value=4)

Mother_Job = st.number_input("Mother Job (0-4)", min_value=0)
Father_Job = st.number_input("Father Job (0-4)", min_value=0)

Reason_for_Choosing_School = st.number_input("Reason for Choosing School (0-3)", min_value=0)
Guardian = st.number_input("Guardian (0-2)", min_value=0)

Travel_Time = st.number_input("Travel Time (1-4)", min_value=1, max_value=4)
Study_Time = st.number_input("Study Time (1-4)", min_value=1, max_value=4)

School_Support = st.number_input("School Support (0/1)", min_value=0)
Family_Support = st.number_input("Family Support (0/1)", min_value=0)
Extra_Paid_Class = st.number_input("Extra Paid Class (0/1)", min_value=0)
Extra_Curricular_Activities = st.number_input("Extra Curricular Activities (0/1)", min_value=0)

Attended_Nursery = st.number_input("Attended Nursery (0/1)", min_value=0)
Wants_Higher_Education = st.number_input("Wants Higher Education (0/1)", min_value=0)
Internet_Access = st.number_input("Internet Access (0/1)", min_value=0)
In_Relationship = st.number_input("In Relationship (0/1)", min_value=0)

Family_Relationship = st.number_input("Family Relationship (1-5)", min_value=1, max_value=5)
Free_Time = st.number_input("Free Time (1-5)", min_value=1, max_value=5)
Going_Out = st.number_input("Going Out (1-5)", min_value=1, max_value=5)

Weekend_Alcohol_Consumption = st.number_input("Weekend Alcohol Consumption (1-5)", min_value=1, max_value=5)
Weekday_Alcohol_Consumption = st.number_input("Weekday Alcohol Consumption (1-5)", min_value=1, max_value=5)

Health_Status = st.number_input("Health Status (1-5)", min_value=1, max_value=5)
Number_of_Absences = st.number_input("Number of Absences", min_value=0)

# ---------- PREDICTION ----------

if st.button("Predict"):

    input_data = pd.DataFrame([[
        School, Gender, Age, Address, Family_Size, Parental_Status,
        Mother_Education, Father_Education, Mother_Job, Father_Job,
        Reason_for_Choosing_School, Guardian, Travel_Time, Study_Time,
        School_Support, Family_Support, Extra_Paid_Class,
        Extra_Curricular_Activities, Attended_Nursery, Wants_Higher_Education,
        Internet_Access, In_Relationship, Family_Relationship, Free_Time,
        Going_Out, Weekend_Alcohol_Consumption, Weekday_Alcohol_Consumption,
        Health_Status, Number_of_Absences
    ]], columns=model.feature_names_in_)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Student is likely to Dropout")
    else:
        st.success("✅ Student will Continue")