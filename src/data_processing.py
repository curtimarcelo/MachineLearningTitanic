import seaborn as sns
import pandas as pd


def load_and_clean_data():
    df = sns.load_dataset("titanic")

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    return df


def clean_and_transform_data(df):
    # Preenche valores ausentes
    df['age'].fillna(df['age'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

    # Remove colunas apenas se existirem no DataFrame
    columns_to_drop = ['name', 'ticket', 'cabin']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Transforma variáveis categóricas em numéricas
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Cria uma nova feature
    df['family_size'] = df['sibsp'] + df['parch']

    return df
