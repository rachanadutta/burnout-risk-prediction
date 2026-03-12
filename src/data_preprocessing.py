import pandas as pd

def preprocess_data():
    df= pd.read_csv("data/raw/student_lifestyle.csv")
    df.dropna(inplace=True)
    df["burnout"]= (df["Stress_Level"] >=6 ).astype(int)
    drop_column_names= ["Student_ID","Stress_Level","Depression"]
    df= df.drop(columns=drop_column_names)
    output_path= "data/processed/clean_student_lifestyle.csv"
    df.to_csv(output_path, index=False)

    print(f"Data preprocessing completed. Cleaned data saved to {output_path}")
    print("preprocessing complete")
    print(df.shape)
if __name__ == "__main__":
    preprocess_data()




