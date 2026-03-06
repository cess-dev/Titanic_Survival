import pandas as pd
import numpy as np

df = pd.read_csv("../data/train_cleaned.csv")

def engineer_features(df):
    # Family size
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
    # Title Extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Deck Extraction - done in data cleaning
    
    # Age Groups
    def age_group(age):
     if age < 13:
        return "Child"
     elif age < 20:
        return "Teen"
     elif age < 60:
        return "Adult"
     else:
        return "Senior"

    df["AgeGroup"] = df["Age"].apply(age_group)

    # Fare per Person
    df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']

    #encoding categorical variables - used one hot encoding
    df = pd.get_dummies(df,
                    columns=["Sex","Embarked","Title","Deck","AgeGroup"],
                    drop_first=True)
    
    # Log Transform for Skew
    df['Fare_Log'] = np.log1p(df['Fare'])

    #Standardization
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scale_cols = ["Age","Fare","Family_Size","Fare_Per_Person"]

    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    return df
if __name__ == "__main__":

    # Load cleaned dataset
    df = pd.read_csv("../data/train_cleaned.csv")

    # Apply feature engineering
    df_features = engineer_features(df)

    # Save new dataset
    df_features.to_csv("../data/train_features.csv", index=False)

    print("Feature engineered dataset saved as train_features.csv")