
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
client.py — Cliente Flower standalone (opcional)
-----------------------------------------------
Permite rodar um cliente único conectando em um servidor Flower.
Obs.: Para simulação com Ray, use analyze.py.
"""
import argparse
import flwr as fl
from sklearn.metrics import accuracy_score

from dataset import load_dataset
from model import create_model

def make_client(csv_path: str, target_col: str = "class"):
    X_train, X_test, y_train, y_test = load_dataset(csv_path, target_col=target_col)
    model = create_model()

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            # Inicializa shapes, se necessário
            try:
                _ = model.coef_
                _ = model.intercept_
            except Exception:
                model.fit(X_train[:2], y_train[:2])
            return [model.coef_, model.intercept_]

        def set_parameters(self, parameters):
            coef, intercept = parameters
            # Garante atributos internos
            try:
                _ = model.n_features_in_
                _ = model.classes_
            except Exception:
                model.fit(X_train[:2], y_train[:2])
            model.coef_ = coef
            model.intercept_ = intercept

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            model.fit(X_train, y_train)
            return self.get_parameters({}), len(X_train), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            return float(1.0 - acc), len(X_test), {"accuracy": float(acc)}

    return FlowerClient()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/ERENO-2.0-100K.csv")
    ap.add_argument("--target-col", type=str, default="class")
    ap.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = ap.parse_args()

    fl.client.start_numpy_client(server_address=args.server, client=make_client(args.csv, args.target_col))

if __name__ == "__main__":
    main()
