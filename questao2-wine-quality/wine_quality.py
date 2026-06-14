"""
Wine Quality — Sistema de Classificação de Qualidade
======================================================
Dataset : winequality-red.csv + winequality-white.csv
          https://archive.ics.uci.edu/dataset/186/wine+quality
Modelos : Random Forest, Gradient Boosting, Regressão Logística
Aluna   : Rebecca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from collections import Counter
from pickle import dump, load
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')


#carregamento e montagem do dataset completo

print("=" * 60)
print("  WINE QUALITY — SISTEMA DE CLASSIFICAÇÃO DE QUALIDADE")
print("=" * 60)

# Carrega os dois arquivos separadamente
red   = pd.read_csv('winequality-red.csv',   sep=';')
white = pd.read_csv('winequality-white.csv', sep=';')

print(f"\nShape vinhos tintos : {red.shape}")
print(f"Shape vinhos brancos: {white.shape}")

# Adiciona coluna indicando o tipo de vinho
red['wine_type']   = 0  # tinto
white['wine_type'] = 1  # branco

# Junta os dois datasets
df = pd.concat([red, white], ignore_index=True)

print(f"\nShape combinado     : {df.shape}")
print(f"\nColunas: {list(df.columns)}")

# análise exploratória


print(f"\nPrimeiras linhas:")
print(df.head())

print(f"\nInformações do dataset:")
print(df.info())

print(f"\nEstatísticas descritivas:")
print(df.describe())

print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

print(f"\nDistribuição da classe (quality):")
print(df['quality'].value_counts().sort_index())
print(df['quality'].value_counts(normalize=True).sort_index().map(lambda x: f"{x:.1%}"))

print(f"\nDistribuição por tipo de vinho:")
print(df['wine_type'].value_counts().rename({0: 'Tinto', 1: 'Branco'}))

# pre-processamento

# Todas as features são contínuas — todas serão normalizadas
# A coluna wine_type (0/1) foi adicionada para distinguir
# tintos de brancos e também será normalizada pois representa
# uma distinção entre grupos, não uma variável binária clínica.

X = df.drop(columns=['quality'])
y = df['quality']

# Escalonamento
scaler = StandardScaler()
X_processado = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

print(f"\n--- DADOS APÓS PRÉ-PROCESSAMENTO ---")
print(X_processado.describe().round(3))

# divisão treino/teste

print(f"\n--- FREQUÊNCIA DAS CLASSES (quality) ---")
pprint(dict(Counter(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X_processado, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTreino : {X_train.shape[0]} amostras")
print(f"Teste  : {X_test.shape[0]} amostras")

# definição dos modelos

modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    ),
    "Regressão Logística": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# treinamento e avaliação

resultados = {}

print("\n" + "=" * 60)
for nome, modelo in modelos.items():
    print(f"\n{'─' * 40}")
    print(f"  Treinando: {nome}")
    print(f"{'─' * 40}")

    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    cm  = confusion_matrix(y_test, y_pred)

    resultados[nome] = {
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'accuracy': acc,
        'f1':       f1,
        'cm':       cm,
        'y_pred':   y_pred,
        'modelo':   modelo,
    }

    print(f"  CV Accuracy (10-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Acurácia (teste)      : {acc:.4f}")
    print(f"  F1-Score (weighted)   : {f1:.4f}")
    print(f"\n  Relatório completo:")
    print(classification_report(y_test, y_pred))

# seleção do melhor modelo

print("\n" + "=" * 60)
print("  SELEÇÃO DO MELHOR MODELO")
print("=" * 60)

melhor_nome = max(resultados, key=lambda n: resultados[n]['f1'])
melhor      = resultados[melhor_nome]

print(f"\n  Modelo selecionado: {melhor_nome}")
print(f"  Acurácia           : {melhor['accuracy']:.4f}")
print(f"  F1-Score           : {melhor['f1']:.4f}")
print(f"  CV Accuracy        : {melhor['cv_mean']:.4f} ± {melhor['cv_std']:.4f}")

print(f"""
  Justificativa:
  O problema de qualidade de vinho é multiclasse (notas 3 a 9)
  com classes desbalanceadas — notas 5 e 6 dominam o dataset.
  O F1-Score weighted foi usado como critério de seleção por
  considerar o desbalanceamento entre as classes.

  O {melhor_nome} apresentou o melhor equilíbrio entre
  acurácia global e desempenho por classe, sendo o mais
  adequado para implantação em um sistema de classificação
  de qualidade de vinhos.
""")

# salvando melhor modelo e scaler

dump(resultados[melhor_nome]['modelo'], open('melhor_modelo_wq.pkl', 'wb'))
dump(scaler,                            open('scaler_wq.pkl', 'wb'))

print(f"  Modelo salvo em: melhor_modelo_wq.pkl")
print(f"  Scaler salvo em: scaler_wq.pkl")

# visualizações

nomes  = list(resultados.keys())
cores  = ['#f4a7b9', '#b5d5f5', '#c3e6cb']
classes_unicas = sorted(y_test.unique())

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f0f0f')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# Acurácia e F1 por modelo
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#1a1a1a')
x = np.arange(len(nomes))
w = 0.3
bars1 = ax1.bar(x - w/2, [resultados[n]['accuracy'] for n in nomes],
                width=w, label='Acurácia', color='#f4a7b9', alpha=0.85, zorder=3)
bars2 = ax1.bar(x + w/2, [resultados[n]['f1'] for n in nomes],
                width=w, label='F1-Score', color='#b5d5f5', alpha=0.85, zorder=3)
for bar in list(bars1) + list(bars2):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8, color='white')
ax1.set_xticks(x)
ax1.set_xticklabels(nomes, color='white', fontsize=9)
ax1.set_ylim(0, 1.1)
ax1.set_title('Acurácia e F1-Score por Modelo', color='white', fontsize=12, pad=10)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#333')
ax1.grid(axis='y', color='#333', linewidth=0.6, zorder=0)
ax1.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#444')

# CV accuracy
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#1a1a1a')
cv_means = [resultados[n]['cv_mean'] for n in nomes]
cv_stds  = [resultados[n]['cv_std']  for n in nomes]
ax2.barh(nomes, cv_means, xerr=cv_stds, color=cores, alpha=0.85,
         error_kw=dict(ecolor='white', lw=1.5, capsize=5), zorder=3)
ax2.set_xlim(0.4, 0.8)
ax2.set_title('CV Accuracy (10-fold ± std)', color='white', fontsize=10, pad=10)
ax2.tick_params(colors='white', labelsize=8)
ax2.spines[:].set_color('#333')
ax2.grid(axis='x', color='#333', linewidth=0.6, zorder=0)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax2.text(m + s + 0.003, i, f'{m:.3f}', va='center', color='white', fontsize=8)

# Matrizes de confusão
for i, (nome, cor) in enumerate(zip(nomes, cores)):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor('#1a1a1a')
    cm = resultados[nome]['cm']
    ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(classes_unicas)))
    ax.set_yticks(range(len(classes_unicas)))
    ax.set_xticklabels(classes_unicas, color='white', fontsize=7)
    ax.set_yticklabels(classes_unicas, color='white', fontsize=7)
    ax.set_xlabel('Previsto', color='white', fontsize=8)
    ax.set_ylabel('Real',     color='white', fontsize=8)
    ax.set_title(f'Conf. Matrix — {nome}', color=cor, fontsize=9, pad=8)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    for r in range(len(classes_unicas)):
        for c in range(len(classes_unicas)):
            ax.text(c, r, cm[r, c], ha='center', va='center',
                    color='white', fontsize=7, fontweight='bold')

plt.suptitle('Wine Quality — Comparação de Classificadores',
             color='white', fontsize=14, y=1.01)
plt.savefig('comparacao_modelos_wq.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print('\nGráfico salvo em: comparacao_modelos_wq.png')


# modulo de inferencia 

def inferir_qualidade(
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
    pH, sulphates, alcohol, wine_type,
    modelo_path='melhor_modelo_wq.pkl',
    scaler_path='scaler_wq.pkl'
):
    """
    Recebe os dados físico-químicos de um vinho e indica
    a qualidade prevista (nota de 3 a 9).
    wine_type: 0 = tinto, 1 = branco
    """
    modelo_inf = load(open(modelo_path, 'rb'))
    scaler_inf = load(open(scaler_path, 'rb'))

    dados = pd.DataFrame([{
        'fixed acidity':        fixed_acidity,
        'volatile acidity':     volatile_acidity,
        'citric acid':          citric_acid,
        'residual sugar':       residual_sugar,
        'chlorides':            chlorides,
        'free sulfur dioxide':  free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density':              density,
        'pH':                   pH,
        'sulphates':            sulphates,
        'alcohol':              alcohol,
        'wine_type':            wine_type,
    }])

    dados_sc     = scaler_inf.transform(dados)
    qualidade    = modelo_inf.predict(dados_sc)[0]
    distribuicao = modelo_inf.predict_proba(dados_sc)[0]
    classes      = modelo_inf.classes_
    score        = max(distribuicao)

    return qualidade, score, distribuicao, classes


def exibir_resultado_inferencia(qualidade, score, distribuicao, classes, wine_type):
    tipo = 'Tinto' if wine_type == 0 else 'Branco'

    print("\n" + "=" * 50)
    print("  RESULTADO DA CLASSIFICAÇÃO DO VINHO")
    print("=" * 50)
    print(f"\n  Tipo          : {tipo}")
    print(f"  Qualidade     : {qualidade} / 10")
    print(f"  Confiança     : {score:.4f} ({score * 100:.1f}%)")
    print(f"\n  Distribuição de probabilidade por nota:")
    for cls, prob in zip(classes, distribuicao):
        barra = '█' * int(prob * 30)
        print(f"    Nota {cls}: {prob:.4f} ({prob * 100:.1f}%) {barra}")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')

    cor_barras = ['#f4a7b9' if c == qualidade else '#b5d5f5' for c in classes]
    bars = ax.bar([str(c) for c in classes], distribuicao,
                  color=cor_barras, alpha=0.85, zorder=3)
    for bar, val in zip(bars, distribuicao):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val * 100:.1f}%', ha='center', va='bottom',
                color='white', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(distribuicao) * 1.25)
    ax.set_xlabel('Nota de Qualidade', color='white', fontsize=11)
    ax.set_ylabel('Probabilidade',     color='white', fontsize=11)
    ax.set_title(f'Wine Quality — Vinho {tipo} | Nota Prevista: {qualidade} (confiança: {score*100:.1f}%)',
                 color='white', fontsize=12, pad=10)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.grid(axis='y', color='#333', linewidth=0.6, zorder=0)

    plt.tight_layout()
    plt.savefig('resultado_inferencia_wq.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print('\n  Gráfico salvo em: resultado_inferencia_wq.png')


#exemplo de inferencia

print("\n" + "=" * 60)
print("  EXEMPLO DE INFERÊNCIA — VINHO HIPOTÉTICO")
print("=" * 60)

qualidade, score, distribuicao, classes = inferir_qualidade(
    fixed_acidity=7.4,
    volatile_acidity=0.70,
    citric_acid=0.00,
    residual_sugar=1.9,
    chlorides=0.076,
    free_sulfur_dioxide=11,
    total_sulfur_dioxide=34,
    density=0.9978,
    pH=3.51,
    sulphates=0.56,
    alcohol=9.4,
    wine_type=0,  # tinto
)

exibir_resultado_inferencia(qualidade, score, distribuicao, classes, wine_type=0)