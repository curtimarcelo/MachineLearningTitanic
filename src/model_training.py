from sklearn.model_selection import train_test_split  # Importação necessária
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def split_data(df):
    features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'family_size']
    target = 'survived'

    x = df[features]
    y = df[target]

    # Divide o conjunto de dados em treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def train_models(X_train, y_train):
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    return log_model, rf_model
