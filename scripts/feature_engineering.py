import pandas as pd
import numpy as np

def engineer_features(df):
    # Derived Features
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
    
    # Title Extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Deck Extraction
    df['Deck'] = df['Cabin'].apply(lambda x: x[0])
    
    # Age Groups
    bins = [0, 12, 19, 59, 100]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Fare per Person
    df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']
    
    # Log Transform for Skew
    df['Fare_Log'] = np.log1p(df['Fare'])
    
    return df