import io
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def _read_csv_robusto(csv_path: str) -> pd.DataFrame:
    """
    Lê um 'CSV' que pode conter linhas de comentário/cabeçalho no estilo ARFF/WEKA
    (ex.: '@relation ...', '@attribute ...', '% ...', '@data').
    Remove essas linhas antes de carregar no pandas.
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # filtra fora linhas que começam com @ ou % (típico de ARFF/WEKA)
    filtered = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("@") or s.startswith("%"):
            # pula linhas de metadados/comentários ARFF
            continue
        filtered.append(ln)

    # Agora tenta ler como CSV "normal"
    buf = io.StringIO("".join(filtered))
    # low_memory=False para evitar DtypeWarning
    df = pd.read_csv(buf, low_memory=False)

    return df

def load_dataset(csv_path="data/ERENO-2.0-100K.csv", target_col="class", test_size=0.2, random_state=42):
    """
    Carrega o dataset a partir de CSV "ruidoso" (pode conter linhas ARFF),
    transforma categóricas em one-hot, padroniza features e retorna X_train, X_test, y_train, y_test.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    # Lê com limpeza de linhas tipo ARFF
    df = _read_csv_robusto(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Coluna de rótulo '{target_col}' não está presente no arquivo {csv_path}.\n"
                         f"Colunas detectadas: {list(df.columns)}")

    # Remove linhas sem rótulo
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Separa X e y
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Converte todas as colunas para numéricas via one-hot (categorias -> dummies)
    # Observação: get_dummies lida com colunas object/strings automaticamente
    X = pd.get_dummies(X, drop_first=False)

    # Converte NAs (se houver) para 0 após one-hot
    X = X.fillna(0)

    # Padroniza todas as colunas numéricas (agora todas são numéricas)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
    )

    return X_train, X_test, y_train, y_test
