import joblib
import pandas as pd

model = joblib.load("models/burnout_model.pkl")


def predict_burnout(input_data: dict):

    df = pd.DataFrame([input_data])

    probability = model.predict_proba(df)[0][1]

    risk_percent = probability * 100

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return risk_percent, risk_level


if __name__ == "__main__":

    sample_input = {
        "Age": 21,
        "CGPA": 3.2,
        "Sleep_Duration": 5,
        "Study_Hours": 8,
        "Social_Media_Hours": 4,
        "Physical_Activity": 1,
        "Gender": "Male",
        "Department": "Engineering"
    }

    risk_percent, risk_level = predict_burnout(sample_input)

    print(f"Burnout Risk: {risk_percent:.2f}%")
    print("Risk Level:", risk_level)