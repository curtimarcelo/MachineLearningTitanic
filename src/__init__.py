from main import X_test, y_test


class TitanicModel:
    def __init__(self, dataset):
        # Inicializa com o dataset
        self.df = dataset
        self.model = None

    def preprocess_data(self):
        # Realiza a limpeza e transformação dos dados
        self.df['age'].fillna(self.df['age'].median(), inplace=True)
        self.df['embarked'].fillna(self.df['embarked'].mode()[0], inplace=True)
        self.df['sex'] = self.df['sex'].map({'male': 0, 'female': 1})
        self.df = self.df.drop(columns=['name', 'ticket', 'cabin'])

    def train_model(self):
        # Treina o modelo
        from sklearn.ensemble import RandomForestClassifier
        X = self.df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
        y = self.df['survived']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self):
        # Avalia o modelo
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)
