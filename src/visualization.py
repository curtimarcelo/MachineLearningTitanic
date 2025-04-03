import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_feature_importance(model, features):

    importances = model.feature_importances_
    indices = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.title("Importância das Variáveis")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel("Importância")
    plt.show()

def plot_survival_by_sex_and_class(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='pclass', hue='sex')
    plt.title("Distribuição de Sobreviventes por Classe e Sexo")