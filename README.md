# Student Performance & Support Prediction Dashboard

## Overview
An end-to-end machine learning project that analyzes student academic and engagement factors and flags students who may require additional academic support.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Random Forest, Streamlit

## Workflow
1. Generate/load structured student data.
2. Perform exploratory analysis.
3. Prepare features and target.
4. Train a Random Forest classification pipeline.
5. Evaluate with Accuracy, Precision, Recall and F1-score.
6. Save the model with Joblib.
7. Deploy an interactive Streamlit dashboard.

## Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Important note
This is an educational portfolio project using synthetic data. The predictions should not be used for real educational decisions without validation, domain review and appropriate privacy safeguards.
