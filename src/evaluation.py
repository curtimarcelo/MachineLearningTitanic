from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(y_test, log_pred, rf_pred):
    log_accuracy = accuracy_score(y_test, log_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print("Acurácia do Modelo de Regressão Logística:", log_accuracy)
    print("Acurácia do Modelo de Random Forest:", rf_accuracy)

    print("\nMatriz de Confusão (Logística):")
    print(confusion_matrix(y_test, log_pred))
    print("\nMatriz de Confusão (Random Forest):")
    print(confusion_matrix(y_test, rf_pred))

    print("\nRelatório de Classificação (Logística):")
    print(classification_report(y_test, log_pred))
    print("\nRelatório de Classificação (Random Forest):")
    print(classification_report(y_test, rf_pred))
