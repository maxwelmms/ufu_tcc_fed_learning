# Federated Learning - Flower Project

Este é um projeto simples de **Aprendizado Federado (FL)** usando o framework **Flower**, desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) para a Especialização em Segurança Cibernética da Universidade Federal de Uberlândia (UFU).  
O objetivo é treinar um modelo de classificação a partir do dataset `ERENO-2.0-100K.csv` e medir o **impacto da técnica de envenenamento de rótulos (Label Poisoning)** na degradação do desempenho do modelo.

Download do dataset [Aqui](https://drive.google.com/file/d/1Il9YL3cOv8ret1NPoDVITSEbwRyaNoRV/view?usp=drivesdk).

---

## 📂 Estrutura
```
UFU_TCC_FED_LEARNING/
│
├─ data/
│   ├─ ERENO-2.0-100K.csv          # Dataset original
│   └─ ERENO-2.0-100K-poisoned.csv # Dataset envenenado (gerado pelo poison.py)
│
├─ src/
│   ├─ client.py                   # Cliente FL
│   ├─ server.py                   # Servidor FL
│   ├─ model.py                    # Definição do modelo
│   ├─ dataset.py                  # Dataset e pré-processamento
│   ├─ poison.py                   # Módulo de envenenamento
│   └─ analyze.py                  # Análise automática (com métricas + gráficos)
│
├─ requirements.txt                # Dependencias libs/modulos
└─ README.md                       # Instruções/documentação
```



Este é um projeto simples de **aprendizado federado** usando o framework **Flower** e um dataset em CSV.



## Requisitos:
```
flwr
scikit-learn
pandas
numpy
matplotlib
```

## Arquivos
- `data/ERENO-2.0-100K.csv` → Dataset original
- `ERENO-2.0-100K-poisoned.csv` → Dataset envenenado
- `src/client.py` → Cliente FL
- `src/server.py` → Servidor FL
- `src/model.py` → Definição do modelo
- `src/dataset.py` → Pré-processamento
- `src/poison.py` → Simulação de **Label Poisoning**
- `src/analyze.py` → Análise automática


## ⚙️ Instalação

1. Clone este repositório ou copie os arquivos.
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Como Rodar
1. Rodar servidor e clientes manualmente (modo federado tradicional)

Inicie o servidor:
```bash
python src/server.py
```

Em dois terminais separados, inicie os clientes:
```bash
python src/client.py
python src/client.py
```

2. Gerar dataset envenenado

Para criar uma versão com label poisoning:
```bash
python src/poison.py
```
* **Isso vai gerar data/ERENO-2.0-100K-poisoned.csv, que pode ser usado no lugar do dataset original para ver o impacto na acurácia e precisão**

3. Executar análise comparativa automática com métricas e gráficos

Execute:
```bash
python src/analyze.py
```

**Isso roda o aprendizado federado duas vezes:**

* 1x com dataset limpo

* 1x com dataset envenenado

**E gera:**

* **Arquivos JSON**

* * results_clean.json → métricas finais do dataset limpo

* * results_poison.json → métricas finais do dataset envenenado

* **Gráficos comparativos**

* * comparison_accuracy.png

* * comparison_precision.png

* * comparison_recall.png

* * comparison_f1.png


## 📊 Interpretação dos Resultados

**JSONs (`results_clean.json`, `results_poison.json`)**

**Exemplo de saída em `results_clean.json`:**

```bash
{
    "dataset": "data/ERENO-2.0-100K.csv",
    "final_accuracy": 0.9123,
    "final_precision": 0.9101,
    "final_recall": 0.9110,
    "final_f1": 0.9105
}
```

* **final_accuracy** → porcentagem de acertos do modelo.

* **final_precision** → capacidade de evitar falsos positivos.

* **final_recall** → capacidade de identificar corretamente os positivos.

* **final_f1** → equilíbrio entre precisão e recall.

## 📌 Compare com o `results_poison.json`.
A diferença entre os dois mostra a degradação causada pelo poisoning.

## Gráficos (comparison_*.png)

Cada gráfico mostra a evolução de uma métrica ao longo dos rounds de FL:

* **Linha Clean** → Treinamento com dataset original.

* **Linha Poisoned** → Treinamento com dataset envenenado.

Se o envenenamento for eficaz, você verá:

* Acurácia caindo.

* Precisão e Recall distorcidos (dependendo de quais classes foram afetadas).

* F1-score reduzido (indicando perda de equilíbrio entre precisão e recall).

## 📌 Conclusão

### 👉 Com este projeto você irá:

- Entender como funciona o aprendizado federado básico com Flower.
- Treinar logistic regression via **federated averaging (FedAvg)**
- Simular ataques de label poisoning.
- Medir quantitativamente o impacto depois do envenenamento usando `Accuracy`, `Precision`, `Recall` e `F1-score`. 
- Visualizar graficamente a degradação de desempenho do modelo.
- Facilmente adaptar para novos experimentos.
