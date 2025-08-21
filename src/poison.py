import pandas as pd
import numpy as np
from typing import Optional

def poison_labels(
    csv_path: str = "data/ERENO-2.0-100K.csv",
    output_path: str = "data/ERENO-2.0-100K-poisoned.csv",
    target_label: str = "class",
    poison_rate: float = 0.10,
    random_state: Optional[int] = 42,
    save_poisoned_indices: bool = True,
):
    """
    Aplica label poisoning em uma fração do dataset, de forma vetorizada.
    - Suporta labels numéricos ou strings.
    - Garante que a nova label seja sempre diferente da original.
    - O(n) e sem loops Python sobre as linhas.

    Estratégia:
    1) Factoriza as classes (mapeia para [0..K-1]).
    2) Seleciona m índices aleatórios (m = poison_rate * N).
    3) Para cada índice selecionado, soma um deslocamento aleatório em [1..K-1] (mod K),
       garantindo que nunca fique igual à classe original.
    4) Mapeia de volta para os rótulos originais e salva o CSV.
    """
    # --- Leitura e checagens básicas ---
    df = pd.read_csv(csv_path, low_memory=False)
    if target_label not in df.columns:
        raise ValueError(f"Coluna '{target_label}' não encontrada em {csv_path}")

    y = df[target_label].values
    # K classes necessárias para trocar o rótulo
    codes, uniques = pd.factorize(y, sort=True)
    K = len(uniques)
    if K < 2:
        raise ValueError("É necessário pelo menos 2 classes para aplicar label poisoning.")

    N = len(df)
    m = max(1, int(poison_rate * N))

    rng = np.random.default_rng(random_state)

    # --- Seleciona índices a serem envenenados ---
    poisoned_idx = rng.choice(N, size=m, replace=False)

    # --- Gera deslocamentos aleatórios em [1..K-1] (garante classe diferente) ---
    # offsets.shape = (m,)
    if K == 2:
        # Caso binário: trocar sempre para a outra classe (offset = 1)
        offsets = np.ones(m, dtype=np.int64)
    else:
        offsets = rng.integers(low=1, high=K, size=m, endpoint=False)  # 1..K-1

    # --- Aplica nova classe (módulo K garante faixa válida) ---
    new_codes = codes.copy()
    new_codes[poisoned_idx] = (codes[poisoned_idx] + offsets) % K

    # --- Reconstrói labels originais a partir dos códigos ---
    poisoned_labels = uniques[new_codes]

    # --- Salva resultado ---
    df_poisoned = df.copy()
    df_poisoned[target_label] = poisoned_labels
    df_poisoned.to_csv(output_path, index=False)

    print(f"[INFO] Dataset envenenado salvo em {output_path}")
    print(f"[INFO] Amostras envenenadas: {m} de {N} ({poison_rate*100:.1f}%) | K={K} classes")

    if save_poisoned_indices:
        # Opcional: salva os índices alterados para auditoria/reprodutibilidade
        np.save(output_path.replace(".csv", "_indices.npy"), poisoned_idx)
        print(f"[INFO] Índices envenenados salvos em {output_path.replace('.csv', '_indices.npy')}")

if __name__ == "__main__":
    poison_labels()
