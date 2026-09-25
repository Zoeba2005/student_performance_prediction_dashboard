import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/student_performance.csv")
features = ["age","study_hours","attendance","previous_score","assignments_completed",
            "sleep_hours","extracurricular_hours","internet_access","parental_support"]
X, y = df[features], df["support_required"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))

joblib.dump((model, features), "models/student_support_model.joblib")
print("Model saved to models/student_support_model.joblib")
