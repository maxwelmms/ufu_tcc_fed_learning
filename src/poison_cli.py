
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
poison_cli.py — Gerador de dataset envenenado (label poisoning) standalone

Uso:
  python poison_cli.py --csv data/ERENO-2.0-100K.csv --poison-rate 0.3 --target-col class
  # Saída padrão: data/ERENO-2.0-100K-poisoned.csv
  # Para definir saída explicitamente:
  python poison_cli.py --csv data/ERENO-2.0-100K.csv --poison-rate 0.3 --target-col class --output data/poisoned.csv

Opções úteis:
  --random-state 42        # reprodutibilidade
  --save-indices           # salva ids das linhas envenenadas (arquivo .indices.json ao lado do CSV de saída)
"""

import argparse
import os
import json

from poison import poison_labels

def main():
    ap = argparse.ArgumentParser(description="Gerar dataset envenenado (label poisoning) a partir de um CSV")
    ap.add_argument("--csv", type=str, required=True, help="Caminho do CSV clean (entrada)")
    ap.add_argument("--output", type=str, default=None, help="Caminho do CSV envenenado (saída). Se omitido, usa <input>-poisoned.csv")
    ap.add_argument("--target-col", type=str, default="class", help="Nome da coluna de rótulo/label")
    ap.add_argument("--poison-rate", type=float, default=0.3, help="Fração do dataset a envenenar [0..1]")
    ap.add_argument("--random-state", type=int, default=42, help="Semente RNG para reprodutibilidade")
    ap.add_argument("--save-indices", action="store_true", help="Salvar ids/índices das linhas envenenadas em JSON")
    args = ap.parse_args()

    input_csv = args.csv
    if args.output is None:
        root, ext = os.path.splitext(input_csv)
        output_csv = root + "-poisoned" + ext
    else:
        output_csv = args.output

    print(f"[POISON] Entrada : {input_csv}")
    print(f"[POISON] Saída   : {output_csv}")
    print(f"[POISON] Taxa    : {args.poison_rate} | target-col: {args.target_col} | seed: {args.random_state}")

    # a função já suporta salvar índices; vamos passar o sinalizador
    poison_labels(
        csv_path=input_csv,
        output_path=output_csv,
        target_label=args.target_col,
        poison_rate=args.poison_rate,
        random_state=args.random_state,
        save_poisoned_indices=args.save_indices,
    )

    if args.save_indices:
        # a função grava <output>.indices.json; apenas informa o caminho previsto:
        idx_path = output_csv + ".indices.json"
        if os.path.exists(idx_path):
            print(f"[POISON] Índices envenenados salvos em: {idx_path}")
        else:
            print("[POISON] Aviso: arquivo de índices não encontrado; verifique permissões/caminho.")

    print("[POISON] Concluído.")

if __name__ == "__main__":
    main()
