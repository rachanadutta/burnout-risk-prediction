import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_model():
    # Load the preprocessed data
    df= pd.read_csv("data/processed/clean_student_lifestyle.csv")
    X= df.drop(columns=["burnout"])
    y= df["burnout"]
    #split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42,stratify=y)
    # Define numeric and categorical features
    numeric_features = [
        "Age",
        "CGPA",
        "Sleep_Duration",
        "Study_Hours",
        "Social_Media_Hours",
        "Physical_Activity"
    ]

    categorical_features = [
        "Gender",
        "Department"
    ]
    #preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(),numeric_features),
            ("cat", OneHotEncoder(),categorical_features)
        ]
    )
    #model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=42
    )
    #pipeline
    pipeline= Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    #train
    pipeline.fit(X_train,y_train)
    #save model
    joblib.dump(pipeline, "models/burnout_model.pkl")

    print("training complete")
    print("model saved to models/burnout_model.pkl")

if __name__ == "__main__":
    train_model()

   
