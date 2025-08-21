from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def create_model():
    # Modelo simples para fins de demonstração
    return LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)

def evaluate_model(model, X_test, y_test):
    # Faz predições e calcula acurácia
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
