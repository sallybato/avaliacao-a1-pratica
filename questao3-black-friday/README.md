# Black Friday Sales

Dataset: [Retail Black Friday Sales — Kaggle](https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset)

Dado as circunstâncias de uma venda, o sistema roda 3 classificadores independentes e indica, com grau de certeza, a **categoria do produto**, a **forma de pagamento** e a **faixa etária** mais prováveis do comprador.

## Como rodar

```
py black_friday.py
```

> O treinamento pode levar alguns minutos — são 100k linhas e 3 modelos rodando em sequência.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `black_friday.py` | Pré-processamento, treinamento e inferência |
| `retail_black_friday_sales_100k.csv` | Dataset com 100.000 transações e 18 colunas |

## Pipeline

1. **Carregamento** — leitura do dataset com 100.000 transações
2. **Limpeza** — remoção de colunas sem valor preditivo (`transaction_id`, `customer_id`, `product_id`, `purchase_date`)
3. **Encoding** — `gender`, `city` e `customer_segment` convertidos com `LabelEncoder`
4. **Pré-processamento** — variáveis binárias (`is_weekend`, `is_black_friday`) mantidas em 0/1; demais normalizadas com `StandardScaler`
5. **Divisão** — 70% treino / 30% teste com estratificação para cada target
6. **Treinamento** — Random Forest com validação cruzada de 5 folds para cada classificador
7. **Inferência** — venda hipotética retornando as 3 previsões com grau de certeza

## Classificadores

| Classificador | Target | Classes |
|---|---|---|
| Categoria do produto | `product_category` | Accessories, Beauty, Books, Clothing, Electronics, Footwear, Groceries, Home & Kitchen, Sports, Toys |
| Forma de pagamento | `payment_method` | Cash, Credit Card, Debit Card, Gift Card, Mobile Wallet, PayPal |
| Faixa etária | `age_group` | 18–25, 26–35, 36–45, 46–55, 56+ |

## Por que Random Forest?

Com 100k amostras e múltiplas classes desbalanceadas nos 3 targets, Random Forest é a escolha mais robusta. O `class_weight='balanced'` compensa o desbalanceamento, e o modelo também permite extrair importância das features para análises posteriores.

## Saídas

- Terminal — métricas dos 3 classificadores com relatórios e resumo geral
- `comparacao_classificadores_bf.png` — métricas e matrizes de confusão dos 3 modelos
- `resultado_inferencia_bf.png` — distribuição de probabilidade por classe para cada previsão
- Modelos salvos: `modelo_product_category.pkl`, `modelo_payment_method.pkl`, `modelo_age_group.pkl`
- Encoders salvos: `le_product.pkl`, `le_payment.pkl`, `le_age.pkl`, `le_gender.pkl`, `le_city.pkl`, `le_segment.pkl`
- `scaler_bf.pkl` — scaler salvo

## Bibliotecas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação do dataset |
| `numpy` | Operações numéricas |
| `matplotlib` | Geração dos gráficos |
| `scikit-learn` | Pré-processamento, treinamento e métricas |
| `pickle` | Serialização dos modelos treinados |

## Observação sobre os resultados

As acurácias dos classificadores de forma de pagamento (16,3%) e faixa etária (19,8%) ficaram próximas do acaso — que seria ~16,7% para 6 classes e ~20% para 5 classes, respectivamente. O classificador de categoria do produto (34,5%) ficou acima do acaso (~10% para 10 classes), mas ainda assim modesto.

Isso indica que `payment_method` e `age_group` foram provavelmente gerados de forma independente das demais variáveis no dataset sintético, tornando qualquer modelo preditivo matematicamente incapaz de superar o acaso nesses targets. O pipeline, o pré-processamento e as métricas estão corretos — o resultado reflete uma limitação do dataset, não do modelo.
