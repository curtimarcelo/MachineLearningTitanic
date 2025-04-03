from src.data_processing import load_and_clean_data, clean_and_transform_data
from src.model_training import split_data, train_models
from src.evaluation import evaluate_model
from src.visualization import plot_feature_importance, plot_survival_by_sex_and_class
from src.prediction import prever_sobrevivencia

df = load_and_clean_data()
df = clean_and_transform_data(df)

X_train, X_test, y_train, y_test = split_data(df)

log_model, rf_model = train_models(X_train, y_train)

log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
evaluate_model(y_test, log_pred, rf_pred)

plot_feature_importance(rf_model, ['pclass', 'sex', 'age', 'family_size', 'fare', 'embarked'])
plot_survival_by_sex_and_class(df)

resultado = prever_sobrevivencia(3, 0, 25, 0, 10, 2, rf_model)
print(resultado)
