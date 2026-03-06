import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def clean_data(df):

    # Handling missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Cabin → Deck
    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("Unknown")

    # Outlier detection
    sns.boxplot(x=df["Fare"])
    plt.show()

    # Cap extreme Fare values
    q_limit = df['Fare'].quantile(0.95)
    df['Fare'] = np.where(df['Fare'] > q_limit, q_limit, df['Fare'])

    # Data consistency
    df['Sex'] = df['Sex'].str.lower().str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    return df


if __name__ == "__main__":

    # Load raw data
    df = pd.read_csv("../data/train.csv")

    # Clean it
    df_cleaned = clean_data(df)

    # Save cleaned dataset
    df_cleaned.to_csv("../data/train_cleaned.csv", index=False)

    print("Cleaned dataset saved as train_cleaned.csv")