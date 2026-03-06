import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

def feature_selection(df):

    # Drop non-numeric / unnecessary columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    
    # Separate features and target
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    #creating a correlation matrix
    corr = df.corr(numeric_only=True)

    #Visualizing it
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False)
    plt.title("Feature Correlation Matrix")
    plt.show()

    model = RandomForestClassifier(random_state=42)

    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print(importances)

    #Visualization
    importances.head(10).plot(kind="barh")
    plt.title("Top Important Features")
    plt.show()

    selected_features = importances.head(10).index.tolist()

    print("Selected features:")
    print(selected_features)

    return X, y
if __name__ == "__main__":

    # Load engineered dataset
    df = pd.read_csv("../data/train_features.csv")

    # Run feature selection
    X, y = feature_selection(df)

    print("Feature selection completed")