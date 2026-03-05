import pandas as pd
import numpy as np

def clean_data(df):
    # Handling missing values
    # Age: Impute with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Impute with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Fill missing with 'U' for Unknown 
    df['Cabin'] = df['Cabin'].fillna('U')
    
    # Capping Fare at the 99th percentile to handle extreme outliers
    q_limit = df['Fare'].quantile(0.99)
    df['Fare'] = np.where(df['Fare'] > q_limit, q_limit, df['Fare'])
    
    # 3. Data Consistency
    df['Sex'] = df['Sex'].str.lower().str.strip()
    df = df.drop_duplicates()
    
    return df