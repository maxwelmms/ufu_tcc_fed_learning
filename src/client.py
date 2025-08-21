import flwr as fl
from sklearn.metrics import accuracy_score
from .dataset import load_dataset
from .model import create_model

# Carrega dados locais
X_train, X_test, y_train, y_test = load_dataset()

# Cria modelo local
model = create_model()
model.fit(X_train, y_train)

# Define client FL
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.coef_, model.intercept_

    def set_parameters(self, parameters):
        coef, intercept = parameters
        model.coef_ = coef
        model.intercept_ = intercept

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.fit(X_train, y_train)
        return self.get_parameters(config={}), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return float(1.0 - acc), len(X_test), {"accuracy": float(acc)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
