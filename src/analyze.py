import json
import sys
import time
import logging
import numpy as np

# Logging bem verboso para você ver o que está acontecendo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze")

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_dataset
from model import create_model


def compute_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    return acc, prec, rec, f1


def plot_metrics(history_clean, history_poison):
    metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in metrics:
        plt.figure()
        plt.plot(history_clean["round"], history_clean[metric], label="Clean", marker="o")
        plt.plot(history_poison["round"], history_poison[metric], label="Poisoned", marker="x")
        plt.title(f"Comparação de {metric.capitalize()} (Clean vs Poisoned)")
        plt.xlabel("Round")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"comparison_{metric}.png")
        plt.close()
        log.info(f"Gráfico salvo: comparison_{metric}.png")


# -------------------------------
#  A) Modo SIMULAÇÃO (requer Ray)
# -------------------------------
def run_federated_simulation(csv_path: str, output_json: str, label: str, num_rounds=5, num_clients=2):
    log.info(f"[SIM] Carregando dados: {csv_path}")
    X_train, X_test, y_train, y_test = load_dataset(csv_path)

    from flwr.simulation import start_simulation
    import flwr as fl

    # histórico para gráficos
    metrics_history = {"round": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    def get_evaluate_fn():
        def evaluate(server_round, parameters, config):
            model = create_model()
            coef, intercept = parameters
            # Ajuste para modelos do sklearn simples (LogReg) — pode precisar checar shapes
            model.coef_, model.intercept_ = coef, intercept
            acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)
            metrics_history["round"].append(server_round)
            metrics_history["accuracy"].append(acc)
            metrics_history["precision"].append(prec)
            metrics_history["recall"].append(rec)
            metrics_history["f1"].append(f1)
            return 1.0 - acc, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        return evaluate

    def client_fn(cid: str):
        import flwr as fl
        model = create_model()

        class Client(fl.client.NumPyClient):
            def get_parameters(self, config):
                # Inicializa coef_ e intercept_ se ainda não existem (sklearn exige após fit)
                try:
                    _ = model.coef_
                    _ = model.intercept_
                except AttributeError:
                    # fit rápido com uma amostra para inicializar as shapes
                    model.fit(X_train[:2], y_train[:2])
                return model.coef_, model.intercept_

            def set_parameters(self, parameters):
                coef, intercept = parameters
                model.coef_, model.intercept_ = coef, intercept

            def fit(self, parameters, config):
                self.set_parameters(parameters)
                model.fit(X_train, y_train)
                return self.get_parameters({}), len(X_train), {}

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)
                return float(1.0 - acc), len(X_test), {
                    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
                }

        return Client()

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=num_clients,
        min_eval_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(),
    )

    log.info(f"[SIM] Iniciando simulação: {num_clients} clientes, {num_rounds} rounds")
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    final_results = {
        "dataset": csv_path,
        "final_accuracy": metrics_history["accuracy"][-1],
        "final_precision": metrics_history["precision"][-1],
        "final_recall": metrics_history["recall"][-1],
        "final_f1": metrics_history["f1"][-1],
    }
    with open(output_json, "w") as f:
        json.dump(final_results, f, indent=4)
    log.info(f"[SIM] Resultados ({label}) salvos em {output_json}: {final_results}")

    return metrics_history, final_results


# ------------------------------------------------
#  B) Fallback CENTRALIZADO (sem Ray/Simulação)
# ------------------------------------------------
def run_centralized(csv_path: str, output_json: str, label: str, rounds=5):
    log.warning(f"[CENTRALIZADO] Rodando fallback sem Ray/Simulação para {label}")
    X_train, X_test, y_train, y_test = load_dataset(csv_path)

    # “Simula” rounds centralizados só para ter curvas
    metrics_history = {"round": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    model = create_model()

    for r in range(1, rounds + 1):
        model.fit(X_train, y_train)
        acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)
        metrics_history["round"].append(r)
        metrics_history["accuracy"].append(acc)
        metrics_history["precision"].append(prec)
        metrics_history["recall"].append(rec)
        metrics_history["f1"].append(f1)
        log.info(f"[CENTRALIZADO] Round {r}: acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

    final_results = {
        "dataset": csv_path,
        "final_accuracy": metrics_history["accuracy"][-1],
        "final_precision": metrics_history["precision"][-1],
        "final_recall": metrics_history["recall"][-1],
        "final_f1": metrics_history["f1"][-1],
    }
    with open(output_json, "w") as f:
        json.dump(final_results, f, indent=4)
    log.info(f"[CENTRALIZADO] Resultados ({label}) salvos em {output_json}: {final_results}")
    return metrics_history, final_results


def main():
    log.info("=== Início da Análise Federada ===")
    start = time.time()

    # Tenta simulação com Ray
    use_fallback = False
    try:
        import flwr  # noqa
        from flwr.simulation import start_simulation  # noqa
        import ray  # noqa
        log.info("[CHECK] flwr e ray encontrados — tentando modo SIMULAÇÃO")
    except Exception as e:
        log.warning(f"[CHECK] Simulação indisponível ({e}). Usando fallback CENTRALIZADO.")
        use_fallback = True

    if not use_fallback:
        try:
            history_clean, clean_results = run_federated_simulation(
                "data/ERENO-2.0-100K.csv", "results_clean.json", label="Clean"
            )
            history_poison, poison_results = run_federated_simulation(
                "data/ERENO-2.0-100K-poisoned.csv", "results_poison.json", label="Poisoned"
            )
        except Exception as e:
            log.error(f"[SIM] Falhou ao simular: {e}")
            log.warning("[SIM] Caindo para modo CENTRALIZADO.")
            use_fallback = True

    if use_fallback:
        history_clean, clean_results = run_centralized(
            "data/ERENO-2.0-100K.csv", "results_clean.json", label="Clean"
        )
        history_poison, poison_results = run_centralized(
            "data/ERENO-2.0-100K-poisoned.csv", "results_poison.json", label="Poisoned"
        )

    # Gráficos e resumo
    plot_metrics(history_clean, history_poison)

    log.info("\n=== Comparação Final ===")
    log.info(f"Acurácia (limpo):    {clean_results['final_accuracy']:.4f}")
    log.info(f"Acurácia (poisoned): {poison_results['final_accuracy']:.4f}")
    log.info(f"Degradação:          {clean_results['final_accuracy'] - poison_results['final_accuracy']:.4f}")

    log.info(f"Concluído em {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
