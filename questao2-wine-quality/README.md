# Wine Quality

Dataset: [Wine Quality — UCI](https://archive.ics.uci.edu/dataset/186/wine+quality)

Dado as propriedades físico-químicas de um vinho, o sistema classifica sua qualidade em uma escala de 3 a 9. O modelo foi treinado com dados de vinhos tintos e brancos combinados.

## Como rodar

```
py wine_quality.py
```

## Arquivos

| Arquivo | Descrição |
|---|---|
| `wine_quality.py` | Pré-processamento, treinamento e inferência |
| `winequality-red.csv` | Dataset de vinhos tintos (1.599 amostras) |
| `winequality-white.csv` | Dataset de vinhos brancos (4.898 amostras) |
| `winequality.names` | Descrição dos atributos |

## Pipeline

1. **Carregamento** — leitura separada dos arquivos de tintos e brancos
2. **Montagem** — os dois datasets são combinados com uma coluna `wine_type` (0 = tinto, 1 = branco)
3. **Pré-processamento** — todas as features são contínuas e normalizadas com `StandardScaler`
4. **Divisão** — 70% treino / 30% teste com estratificação por `quality`
5. **Treinamento** — 3 modelos avaliados com validação cruzada de 10 folds
6. **Seleção** — modelo com melhor F1-Score weighted é escolhido para implantação
7. **Inferência** — vinho hipotético com atributos físico-químicos de entrada

## Modelos avaliados

| Modelo | Por que foi incluído |
|---|---|
| Random Forest | Robusto ao desbalanceamento de classes |
| Gradient Boosting | Aprende iterativamente os erros anteriores |
| Regressão Logística | Baseline linear com boa interpretabilidade |

O critério de seleção é o F1-Score weighted — isso importa porque notas 5 e 6 dominam o dataset enquanto as extremas (3, 4, 8, 9) são raras.

## Saídas

- Terminal — métricas dos 3 modelos, matrizes de confusão e relatórios
- `comparacao_modelos_wq.png` — comparativo dos 3 modelos com matrizes de confusão
- `resultado_inferencia_wq.png` — distribuição de probabilidade por nota do vinho hipotético
- `melhor_modelo_wq.pkl` e `scaler_wq.pkl` — modelo e scaler salvos

## Bibliotecas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação e combinação dos datasets |
| `numpy` | Operações numéricas |
| `matplotlib` | Geração dos gráficos |
| `scikit-learn` | Pré-processamento, treinamento e métricas |
| `pickle` | Serialização do modelo treinado |
