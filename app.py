import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("Student Performance & Support Dashboard")

df = pd.read_csv("data/student_performance.csv")
model, features = joblib.load("models/student_support_model.joblib")

st.sidebar.header("Student Prediction")
age = st.sidebar.slider("Age",17,22,20)
study = st.sidebar.slider("Study hours/day",1.0,8.0,4.0)
attendance = st.sidebar.slider("Attendance %",55.0,100.0,80.0)
previous = st.sidebar.slider("Previous score",40.0,95.0,70.0)
assignments = st.sidebar.slider("Assignments completed",3,10,7)
sleep = st.sidebar.slider("Sleep hours",4.0,9.0,7.0)
extra = st.sidebar.slider("Extracurricular hours",0.0,4.0,1.0)
internet = st.sidebar.selectbox("Internet access", [0,1], format_func=lambda x: "Yes" if x else "No")
support = st.sidebar.slider("Parental support",1,5,3)

row = pd.DataFrame([[age,study,attendance,previous,assignments,sleep,extra,internet,support]], columns=features)
prediction = model.predict(row)[0]

c1,c2,c3 = st.columns(3)
c1.metric("Students", len(df))
c2.metric("Average Score", f"{df.final_score.mean():.1f}")
c3.metric("Average Attendance", f"{df.attendance.mean():.1f}%")

st.subheader("Performance Distribution")
st.bar_chart(df["performance"].value_counts())

st.subheader("Attendance vs Final Score")
st.scatter_chart(df[["attendance","final_score"]].rename(columns={"attendance":"Attendance","final_score":"Final Score"}))

st.subheader("Individual Support Prediction")
if prediction == 1:
    st.warning("Prediction: Student may require additional academic support.")
else:
    st.success("Prediction: Student is not currently flagged for additional support.")

st.dataframe(df.head(20), use_container_width=True)
