
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze.py — Simulação FL (Flower 1.20) com Ray e comparação Clean vs Poisoned
-----------------------------------------------------------------------------
- Aceita parâmetros de execução: rounds, clients, dataset base, etc.
- Gera/usa dataset envenenado automaticamente (poison_rate ajustável).
- Roda em modo SIMULAÇÃO com Ray (quando disponível); cai para modo CENTRALIZADO se faltar Ray.
- Salva métricas por round e o resumo final em JSON (results_clean.json / results_poison.json).
- Gera gráficos PNG comparando Clean vs Poisoned (accuracy/precision/recall/F1).
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_dataset
from model import create_model
from poison import poison_labels

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyze")

# ----------------------------------------------------------------------------
# Utilidades de métricas e plots
# ----------------------------------------------------------------------------
def compute_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    return acc, prec, rec, f1

def plot_metrics(history_clean: Dict[str, List[float]], history_poison: Dict[str, List[float]]):
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
        fname = f"comparison_{metric}.png"
        plt.savefig(fname, dpi=160)
        plt.close()
        log.info(f"Gráfico salvo: {fname}")

def save_json_per_round(output_json: str, csv_path: str, setup: dict, history: Dict[str, List[float]]):
    """Salva JSON com métricas por round + final/total."""
    rounds = history["round"]
    by_round = []
    for i in range(len(rounds)):
        by_round.append({
            "round": int(history["round"][i]),
            "accuracy": float(history["accuracy"][i]),
            "precision": float(history["precision"][i]),
            "recall": float(history["recall"][i]),
            "f1": float(history["f1"][i]),
        })
    final = by_round[-1] if by_round else {}

    payload = {
        "dataset": csv_path,
        "setup": setup,
        "by_round": by_round,
        "final": final,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    log.info(f"JSON salvo: {output_json}")

# ----------------------------------------------------------------------------
# Helpers para parâmetros do modelo (sklearn LogisticRegression)
# ----------------------------------------------------------------------------
def _ensure_model_initialized(model, X_boot, y_boot):
    """Faz um fit rápido só para inicializar n_features_in_, classes_, etc."""
    try:
        _ = model.coef_
        _ = model.intercept_
        _ = model.n_features_in_
        _ = model.classes_
    except Exception:
        # usa algumas amostras para inicializar atributos internos
        n = min(len(X_boot), max(2, len(np.unique(y_boot))))
        model.fit(X_boot[:n], y_boot[:n])

def get_model_params(model) -> List[np.ndarray]:
    """Extrai parâmetros como lista de ndarrays (formato esperado pelo Flower NumPyClient)."""
    return [model.coef_.copy(), model.intercept_.copy()]

def set_model_params(model, params: List[np.ndarray], X_boot, y_boot):
    """Configura coef_ e intercept_ garantindo que os atributos existam."""
    _ensure_model_initialized(model, X_boot, y_boot)
    coef, intercept = params
    model.coef_ = np.array(coef, copy=True)
    model.intercept_ = np.array(intercept, copy=True)

# ----------------------------------------------------------------------------
# SIMULAÇÃO com Ray
# ----------------------------------------------------------------------------
def run_federated_simulation(csv_path: str, output_json: str, label: str,
                             num_rounds: int, num_clients: int,
                             test_size: float, target_col: str) -> Tuple[Dict[str, List[float]], dict]:
    log.info(f"[SIM] Carregando dados: {csv_path}")
    X_train, X_test, y_train, y_test = load_dataset(csv_path, target_col=target_col, test_size=test_size)

    # classes globais (garante shape consistente de coef_)
    global_classes = np.unique(y_train)
    # pega 1 exemplo por classe para aquecimento/augment
    class_anchor_idx = []
    for c in global_classes:
        idx = np.where(np.array(y_train) == c)[0]
        class_anchor_idx.append(idx[0])
    class_anchor_idx = np.array(class_anchor_idx, dtype=int)

    # partição simples (iid aproximado)
    indices = np.arange(len(X_train))
    shards = np.array_split(indices, num_clients)

    # amostras para bootstrap (inicialização do modelo) — inclui pelo menos 1 por classe
    X_boot = np.vstack([X_train[class_anchor_idx], X_train[:max(2, len(global_classes))]])
    y_boot = np.concatenate([np.array(y_train)[class_anchor_idx], np.array(y_train)[:max(2, len(global_classes))]])

    # histórico para gráficos/JSON
    history = {"round": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    # Criar client_fn
    def client_fn(cid: str):
        import flwr as fl
        cid_int = int(cid)
        client_idx = shards[cid_int]
        X_tr = X_train[client_idx]
        y_tr = np.array(y_train)[client_idx]
        # garante presença de todas as classes no treino local
        X_tr = np.vstack([X_tr, X_train[class_anchor_idx]])
        y_tr = np.concatenate([y_tr, np.array(y_train)[class_anchor_idx]])

        model = create_model()

        class Client(fl.client.NumPyClient):
            def get_parameters(self, config):
                _ensure_model_initialized(model, X_boot, y_boot)
                return get_model_params(model)

            def fit(self, parameters, config):
                set_model_params(model, parameters, X_boot, y_boot)
                model.fit(X_tr, y_tr)
                return get_model_params(model), len(X_tr), {}

            def evaluate(self, parameters, config):
                set_model_params(model, parameters, X_boot, y_boot)
                acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)
                # Flower espera (loss, num_examples, metrics)
                return float(1.0 - acc), len(X_test), {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }

        return Client()

    # Strategy com evaluate_fn no servidor para registrar métricas a cada round
    import flwr as fl

    def get_evaluate_fn():
        def evaluate(server_round: int, parameters: List[np.ndarray], config):
            # Avalia o modelo agregado no conjunto de teste global
            model = create_model()
            set_model_params(model, parameters, X_boot, y_boot)
            acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)

            history["round"].append(server_round)
            history["accuracy"].append(acc)
            history["precision"].append(prec)
            history["recall"].append(rec)
            history["f1"].append(f1)

            return float(1.0 - acc), {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
        return evaluate

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(),
    )

    # Inicializa Ray explicitamente (melhor em Windows)
    try:
        import ray
        if not ray.is_initialized():
            # num_cpus=None deixa Ray detectar automaticamente
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        client_resources = {"num_cpus": 1}
    except Exception as e:
        log.warning(f"[SIM] Ray não pôde inicializar ({e}). Continuando sem especificar recursos.")
        client_resources = None

    from flwr.simulation import start_simulation

    log.info(f"[SIM] Iniciando simulação: clients={num_clients}, rounds={num_rounds}")
    start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    setup = {"mode": "simulation", "clients": num_clients, "rounds": num_rounds, "target_col": target_col}
    save_json_per_round(output_json, csv_path, setup, history)

    final = {
        "dataset": csv_path,
        "final_accuracy": float(history["accuracy"][-1]),
        "final_precision": float(history["precision"][-1]),
        "final_recall": float(history["recall"][-1]),
        "final_f1": float(history["f1"][-1]),
    }
    log.info(f"[SIM] Final ({label}): {final}")
    return history, final

# ----------------------------------------------------------------------------
# Fallback CENTRALIZADO (sem Ray/Simulação)
# ----------------------------------------------------------------------------
def run_centralized(csv_path: str, output_json: str, label: str,
                    rounds: int, test_size: float, target_col: str) -> Tuple[Dict[str, List[float]], dict]:
    log.warning(f"[CENTRALIZADO] Sem Ray — executando modo simples para {label}")
    X_train, X_test, y_train, y_test = load_dataset(csv_path, target_col=target_col, test_size=test_size)

    history = {"round": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    model = create_model()

    for r in range(1, rounds + 1):
        model.fit(X_train, y_train)
        acc, prec, rec, f1 = compute_metrics(model, X_test, y_test)
        history["round"].append(r)
        history["accuracy"].append(acc)
        history["precision"].append(prec)
        history["recall"].append(rec)
        history["f1"].append(f1)
        log.info(f"[CENTRALIZADO] Round {r}: acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}")

    setup = {"mode": "centralized", "clients": 1, "rounds": rounds, "target_col": target_col}
    save_json_per_round(output_json, csv_path, setup, history)

    final = {
        "dataset": csv_path,
        "final_accuracy": float(history["accuracy"][-1]),
        "final_precision": float(history["precision"][-1]),
        "final_recall": float(history["recall"][-1]),
        "final_f1": float(history["f1"][-1]),
    }
    return history, final

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Flower 1.20 + Ray — análise Clean vs Poisoned")
    p.add_argument("--csv", type=str, default="data/ERENO-2.0-100K.csv", help="Caminho do CSV base (clean)")
    p.add_argument("--target-col", type=str, default="class", help="Nome da coluna de rótulo/label")
    p.add_argument("--poison-rate", type=float, default=0.30, help="Fração a envenenar (0..1)")
    p.add_argument("--clients", type=int, default=4, help="Número de clientes na simulação")
    p.add_argument("--rounds", type=int, default=5, help="Número de rounds")
    p.add_argument("--test-size", type=float, default=0.2, help="Proporção para teste (0..1)")
    p.add_argument("--force-central", action="store_true", help="Força modo centralizado (sem Ray)")
    p.add_argument("--regen-poison", action="store_true", help="Recria o CSV envenenado mesmo se já existir")
    return p.parse_args()

def main():
    args = parse_args()
    start = time.time()
    log.info("=== Início da Análise Federada (Flower 1.20) ===")
    log.info(f"Parâmetros: {vars(args)}")

    base_csv = args.csv
    target_col = args.target_col

    # Caminho do dataset envenenado
    root, ext = os.path.splitext(base_csv)
    poison_csv = root + "-poisoned" + ext

    # Gera o CSV envenenado se necessário
    if args.regen_poison or (not os.path.exists(poison_csv)):
        log.info(f"[POISON] Gerando dataset envenenado em '{poison_csv}' (rate={args.poison_rate:.2f})")
        poison_labels(csv_path=base_csv, output_path=poison_csv, target_label=target_col, poison_rate=args.poison_rate, random_state=42, save_poisoned_indices=True)

    # Decide modo com Ray
    use_central = args.force_central
    if not use_central:
        try:
            import flwr  # noqa: F401
            from flwr.simulation import start_simulation  # noqa: F401
            import ray  # noqa: F401
            log.info("[CHECK] flwr e ray encontrados — usando modo SIMULAÇÃO")
        except Exception as e:
            log.warning(f"[CHECK] Simulação indisponível ({e}). Usando fallback CENTRALIZADO.")
            use_central = True

    # Executa Clean/Poison
    if not use_central:
        hist_clean, res_clean = run_federated_simulation(
            base_csv, "results_clean.json", "Clean",
            num_rounds=args.rounds, num_clients=args.clients,
            test_size=args.test_size, target_col=target_col
        )
        hist_po, res_po = run_federated_simulation(
            poison_csv, "results_poison.json", "Poisoned",
            num_rounds=args.rounds, num_clients=args.clients,
            test_size=args.test_size, target_col=target_col
        )
    else:
        hist_clean, res_clean = run_centralized(
            base_csv, "results_clean.json", "Clean",
            rounds=args.rounds, test_size=args.test_size, target_col=target_col
        )
        hist_po, res_po = run_centralized(
            poison_csv, "results_poison.json", "Poisoned",
            rounds=args.rounds, test_size=args.test_size, target_col=target_col
        )

    # Plota comparações
    plot_metrics(hist_clean, hist_po)

    # Log de comparação
    log.info("\n=== Comparação Final ===")
    log.info(f"Acurácia (Clean):   {res_clean['final_accuracy']:.4f}")
    log.info(f"Acurácia (Poison):  {res_po['final_accuracy']:.4f}")
    log.info(f"Degradação (Δacc):  {res_clean['final_accuracy'] - res_po['final_accuracy']:.4f}")

    log.info(f"Concluído em {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
