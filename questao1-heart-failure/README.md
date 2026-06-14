# Heart Failure

Dataset: [Heart Failure Clinical Records — UCI](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

Dado os dados clínicos de um paciente, o sistema indica se ele pertence ao grupo **Sobreviveu** ou **Risco de Óbito**, com base em um modelo treinado em 299 registros reais de cardiologia.

## Como rodar

```
py heart_failure.py
```

## Arquivos

| Arquivo | Descrição |
|---|---|
| `heart_failure.py` | Pré-processamento, treinamento e inferência |
| `heart_failure_clinical_records_dataset.csv` | Dataset original do UCI |

## Pipeline

1. **Carregamento** — leitura do CSV com 299 pacientes e 13 atributos clínicos
2. **Pré-processamento** — variáveis binárias (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`) são mantidas em 0/1; variáveis contínuas são normalizadas com `StandardScaler`
3. **Divisão** — 70% treino / 30% teste com estratificação
4. **Treinamento** — Random Forest com validação cruzada de 10 folds
5. **Avaliação** — acurácia, precisão, recall, F1-Score, ROC-AUC, especificidade e sensibilidade
6. **Inferência** — paciente hipotético com atributos de entrada reais

## Por que Random Forest?

Dataset pequeno (299 amostras), mix de variáveis binárias e contínuas, e classes desbalanceadas — Random Forest lida bem com os três. O parâmetro `class_weight='balanced'` compensa o desbalanceamento, e o modelo dispensa que todas as variáveis estejam na mesma escala.

## Saídas

- Terminal — métricas, matriz de confusão e relatório de classificação
- `resultado_modelo_hf.png` — métricas, matriz de confusão e curva ROC
- `resultado_inferencia_hf.png` — distribuição de probabilidade do paciente hipotético
- `melhor_modelo_hf.pkl` e `scaler_hf.pkl` — modelo e scaler salvos

## Bibliotecas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação do dataset |
| `numpy` | Operações numéricas |
| `matplotlib` | Geração dos gráficos |
| `scikit-learn` | Pré-processamento, treinamento e métricas |
| `pickle` | Serialização do modelo treinado |
