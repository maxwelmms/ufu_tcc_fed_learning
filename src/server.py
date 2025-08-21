import flwr as fl

# Estratégia padrão de FedAvg
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_eval_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    fl.server.start_server(server_address="127.0.0.1:8080", config={"num_rounds": 3}, strategy=strategy)
