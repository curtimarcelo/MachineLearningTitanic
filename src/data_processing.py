import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_and_clean_data():
    df = sns.load_dataset("titanic")
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    return df


def clean_and_transform_data(df):
    df['age'] = df['age'].fillna(df['age'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    columns_to_drop = ['name', 'ticket', 'cabin']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['family_size'] = df['sibsp'] + df['parch']
    return df


def train_and_predict_model(df):
    X = df.drop(columns=['survived'])
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    X_test = pd.DataFrame(X_test, columns=X_train.columns)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    return model


if __name__ == "__main__":
    df = load_and_clean_data()
    df = clean_and_transform_data(df)
    model = train_and_predict_model(df)
