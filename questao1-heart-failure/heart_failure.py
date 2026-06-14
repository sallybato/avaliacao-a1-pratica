"""
Heart Failure — Sistema de Classificação de Risco
===================================================
Dataset : heart_failure_clinical_records_dataset.csv
          https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
Modelos : Random Forest, Gradient Boosting, Regressão Logística
Autor   : Rebecca
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
    confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score,
    RocCurveDisplay
)
from collections import Counter
from pickle import dump, load
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ============================================================

print("=" * 60)
print("  HEART FAILURE — SISTEMA DE CLASSIFICAÇÃO DE RISCO")
print("=" * 60)

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

print(f"\nShape original: {df.shape}")
print("\nPrimeiras linhas:")
print(df.head())
print("\nInformações do dataset:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe())

# ============================================================
# ANÁLISE DAS COLUNAS BINÁRIAS
# ============================================================

# O dataset possui colunas binárias (0/1) que representam
# categorias e NÃO devem ser normalizadas pelo StandardScaler,
# pois já estão na escala correta e a normalização distorceria
# seu significado semântico (ex: anaemia = 0 ou 1).
#
# Colunas binárias identificadas:
#   - anaemia         (0 = não, 1 = sim)
#   - diabetes        (0 = não, 1 = sim)
#   - high_blood_pressure (0 = não, 1 = sim)
#   - sex             (0 = feminino, 1 = masculino)
#   - smoking         (0 = não, 1 = sim)
#
# Estratégia: escalonar APENAS as colunas contínuas.

colunas_binarias  = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
colunas_continuas = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                     'platelets', 'serum_creatinine', 'serum_sodium', 'time']

print(f"\n### COLUNAS BINÁRIAS (não serão normalizadas) ###")
print(colunas_binarias)
print(f"\n### COLUNAS CONTÍNUAS (serão normalizadas) ###")
print(colunas_continuas)

# Verificar valores nulos
print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

# Distribuição do target
print(f"\nDistribuição da classe (DEATH_EVENT):")
print(df['DEATH_EVENT'].value_counts())
print(df['DEATH_EVENT'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

# ============================================================
# SEPARAÇÃO DE ATRIBUTOS E TARGET
# ============================================================

X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# ============================================================
# PRÉ-PROCESSAMENTO — ESCALONAMENTO SELETIVO
# ============================================================

# Fit do scaler apenas nas colunas contínuas
scaler = StandardScaler()
X_processado = X.copy()
X_processado[colunas_continuas] = scaler.fit_transform(X[colunas_continuas])

print(f"\n### DADOS APÓS PRÉ-PROCESSAMENTO ###")
print(X_processado.describe().round(3))

# ============================================================
# DIVISÃO TREINO / TESTE
# ============================================================

print(f"\n### FREQUÊNCIA DAS CLASSES ###")
pprint(dict(Counter(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X_processado, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTreino : {X_train.shape[0]} amostras")
print(f"Teste  : {X_test.shape[0]} amostras")

# ============================================================
# DEFINIÇÃO DOS MODELOS
# ============================================================

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

# ============================================================
# TREINAMENTO E AVALIAÇÃO
# ============================================================

resultados = {}

print("\n" + "=" * 60)
for nome, modelo in modelos.items():
    print(f"\n{'─' * 40}")
    print(f"  Treinando: {nome}")
    print(f"{'─' * 40}")

    # Validação cruzada
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')

    # Treino final
    modelo.fit(X_train, y_train)
    y_pred   = modelo.predict(X_test)
    y_proba  = modelo.predict_proba(X_test)[:, 1]

    # Métricas
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)

    # Especificidade e sensibilidade
    tn, fp, fn, tp = cm.ravel()
    especificidade = tn / (tn + fp)
    sensibilidade  = tp / (tp + fn)

    resultados[nome] = {
        'cv_mean':        cv_scores.mean(),
        'cv_std':         cv_scores.std(),
        'accuracy':       acc,
        'precision':      prec,
        'recall':         rec,
        'f1':             f1,
        'roc_auc':        auc,
        'especificidade': especificidade,
        'sensibilidade':  sensibilidade,
        'cm':             cm,
        'y_pred':         y_pred,
        'y_proba':        y_proba,
        'modelo':         modelo,
    }

    print(f"  CV Accuracy (10-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Acurácia (teste)      : {acc:.4f}")
    print(f"  Precision             : {prec:.4f}")
    print(f"  Recall                : {rec:.4f}")
    print(f"  F1-Score              : {f1:.4f}")
    print(f"  ROC-AUC               : {auc:.4f}")
    print(f"  Especificidade        : {especificidade:.4f}")
    print(f"  Sensibilidade         : {sensibilidade:.4f}")
    print(f"\n  Matriz de Confusão:\n{cm}")
    print(f"\n  Relatório completo:")
    print(classification_report(y_test, y_pred,
                                target_names=['Sobreviveu', 'Faleceu']))

# ============================================================
# SELEÇÃO DO MELHOR MODELO
# ============================================================

print("\n" + "=" * 60)
print("  SELEÇÃO DO MELHOR MODELO")
print("=" * 60)

melhor_nome = max(resultados, key=lambda n: resultados[n]['roc_auc'])
melhor      = resultados[melhor_nome]

print(f"\n  Modelo selecionado: {melhor_nome}")
print(f"  ROC-AUC            : {melhor['roc_auc']:.4f}")
print(f"  F1-Score           : {melhor['f1']:.4f}")
print(f"  Sensibilidade      : {melhor['sensibilidade']:.4f}")
print(f"  Especificidade     : {melhor['especificidade']:.4f}")

print(f"""
  Justificativa:
  O metaestimador {melhor_nome} foi selecionado por apresentar
  o melhor ROC-AUC, que mede a capacidade do modelo de separar
  pacientes sobreviventes de pacientes em risco de óbito.

  Em contexto clínico, falsos negativos (classificar como
  saudável um paciente que vai a óbito) são mais custosos
  que falsos positivos. Por isso, além do ROC-AUC, a
  sensibilidade é uma métrica crítica neste problema.

  O tratamento diferenciado das variáveis binárias (anaemia,
  diabetes, high_blood_pressure, sex, smoking) — mantidas em
  sua escala original 0/1 sem normalização — preserva seu
  significado semântico e evita distorções no aprendizado.
""")

# ============================================================
# SALVANDO MELHOR MODELO E SCALER
# ============================================================

dump(resultados[melhor_nome]['modelo'], open('melhor_modelo_hf.pkl', 'wb'))
dump(scaler,                            open('scaler_hf.pkl', 'wb'))

print(f"  Modelo salvo em: melhor_modelo_hf.pkl")
print(f"  Scaler salvo em: scaler_hf.pkl")

# ============================================================
# VISUALIZAÇÕES
# ============================================================

nomes    = list(resultados.keys())
cores    = ['#f4a7b9', '#b5d5f5', '#c3e6cb']
metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
labels   = ['Acurácia', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor('#0f0f0f')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Gráfico de barras agrupadas
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor('#1a1a1a')
x = np.arange(len(metricas))
w = 0.25

for i, (nome, cor) in enumerate(zip(nomes, cores)):
    vals = [resultados[nome][m] for m in metricas]
    bars = ax1.bar(x + i * w, vals, width=w, label=nome, color=cor, alpha=0.85, zorder=3)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7.5, color='white')

ax1.set_xticks(x + w)
ax1.set_xticklabels(labels, color='white')
ax1.set_ylim(0, 1.12)
ax1.set_title('Métricas por Modelo', color='white', fontsize=13, pad=12)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#333')
ax1.grid(axis='y', color='#333', linewidth=0.6, zorder=0)
ax1.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#444')

# CV accuracy com desvio padrão
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor('#1a1a1a')
cv_means = [resultados[n]['cv_mean'] for n in nomes]
cv_stds  = [resultados[n]['cv_std']  for n in nomes]
ax2.barh(nomes, cv_means, xerr=cv_stds, color=cores, alpha=0.85,
         error_kw=dict(ecolor='white', lw=1.5, capsize=5), zorder=3)
ax2.set_xlim(0.5, 1.0)
ax2.set_title('CV Accuracy (10-fold ± std)', color='white', fontsize=11, pad=10)
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#333')
ax2.grid(axis='x', color='#333', linewidth=0.6, zorder=0)
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax2.text(m + s + 0.003, i, f'{m:.3f}', va='center', color='white', fontsize=8.5)

# Matrizes de confusão
for i, (nome, cor) in enumerate(zip(nomes, cores)):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor('#1a1a1a')
    cm = resultados[nome]['cm']
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Sobreviveu', 'Faleceu'], color='white', fontsize=8)
    ax.set_yticklabels(['Sobreviveu', 'Faleceu'], color='white', fontsize=8)
    ax.set_xlabel('Previsto', color='white', fontsize=9)
    ax.set_ylabel('Real',     color='white', fontsize=9)
    ax.set_title(f'Conf. Matrix — {nome}', color=cor, fontsize=9, pad=8)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    for r in range(2):
        for c in range(2):
            ax.text(c, r, cm[r, c], ha='center', va='center',
                    color='white', fontsize=13, fontweight='bold')

plt.suptitle('Heart Failure — Comparação de Classificadores',
             color='white', fontsize=14, y=1.01)
plt.savefig('comparacao_modelos_hf.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print('\nGráfico salvo em: comparacao_modelos_hf.png')

# Curvas ROC
fig2, ax = plt.subplots(figsize=(7, 6))
fig2.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

for nome, cor in zip(nomes, cores):
    RocCurveDisplay.from_predictions(
        y_test, resultados[nome]['y_proba'],
        name=f"{nome} (AUC={resultados[nome]['roc_auc']:.3f})",
        ax=ax, color=cor
    )

ax.plot([0, 1], [0, 1], 'w--', lw=0.8, label='Baseline')
ax.set_title('Curvas ROC — Heart Failure', color='white', fontsize=13)
ax.tick_params(colors='white')
ax.spines[:].set_color('#444')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#555')
ax.grid(color='#333', linewidth=0.5)

plt.tight_layout()
plt.savefig('curvas_roc_hf.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.show()
print('Curvas ROC salvas em: curvas_roc_hf.png')


# ============================================================
# MÓDULO DE INFERÊNCIA
# ============================================================

def inferir_paciente(
    age, anaemia, creatinine_phosphokinase, diabetes,
    ejection_fraction, high_blood_pressure, platelets,
    serum_creatinine, serum_sodium, sex, smoking, time,
    modelo_path='melhor_modelo_hf.pkl',
    scaler_path='scaler_hf.pkl'
):
    """
    Recebe os dados de um paciente desconhecido e indica
    a qual grupo ele pertence (Sobreviveu / Faleceu).
    """
    modelo_inf = load(open(modelo_path, 'rb'))
    scaler_inf = load(open(scaler_path, 'rb'))

    # Monta o vetor na mesma ordem do treino
    dados = pd.DataFrame([{
        'age':                      age,
        'anaemia':                  anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes':                 diabetes,
        'ejection_fraction':        ejection_fraction,
        'high_blood_pressure':      high_blood_pressure,
        'platelets':                platelets,
        'serum_creatinine':         serum_creatinine,
        'serum_sodium':             serum_sodium,
        'sex':                      sex,
        'smoking':                  smoking,
        'time':                     time,
    }])

    # Escalonar apenas as colunas contínuas
    dados[colunas_continuas] = scaler_inf.transform(dados[colunas_continuas])

    # Predição
    classe       = modelo_inf.predict(dados)[0]
    distribuicao = modelo_inf.predict_proba(dados)[0]
    score        = distribuicao[1]

    resultado = 'RISCO DE ÓBITO' if classe == 1 else 'SOBREVIVEU'
    return resultado, score, distribuicao


def exibir_resultado_inferencia(resultado, score, distribuicao):
    print("\n" + "=" * 50)
    print("  RESULTADO DA ANÁLISE DO PACIENTE")
    print("=" * 50)
    print(f"\n  Classificação : {resultado}")
    print(f"  Score de risco: {score:.4f} ({score * 100:.1f}%)")
    print(f"\n  Distribuição de probabilidade:")
    print(f"    Sobreviveu    : {distribuicao[0]:.4f} ({distribuicao[0] * 100:.1f}%)")
    print(f"    Risco de óbito: {distribuicao[1]:.4f} ({distribuicao[1] * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0f0f0f')

    # Barras de probabilidade
    ax = axes[0]
    ax.set_facecolor('#1a1a1a')
    classes    = ['Sobreviveu', 'Risco de Óbito']
    cores_dist = ['#c3e6cb', '#f4a7b9']
    bars = ax.bar(classes, distribuicao, color=cores_dist, alpha=0.85, width=0.5, zorder=3)
    for bar, val in zip(bars, distribuicao):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val * 100:.1f}%', ha='center', va='bottom',
                color='white', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title('Distribuição de Probabilidade', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    ax.grid(axis='y', color='#333', linewidth=0.6, zorder=0)

    # Gauge de risco
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')
    cor_score = '#f4a7b9' if score >= 0.5 else '#c3e6cb'
    ax2.barh(['Risco de\nÓbito'], [score],
             color=cor_score, alpha=0.85, height=0.4, zorder=3)
    ax2.barh(['Risco de\nÓbito'], [1 - score],
             left=[score], color='#333', alpha=0.5, height=0.4, zorder=2)
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color='white', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(0.5, 0.65, 'threshold 0.5', ha='center', color='white',
             fontsize=8, alpha=0.6, transform=ax2.get_xaxis_transform())
    ax2.text(score, 0, f' {score * 100:.1f}%', va='center',
             color='white', fontsize=13, fontweight='bold')
    ax2.set_title(f'Score: {resultado}', color=cor_score, fontsize=12)
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#333')
    ax2.grid(axis='x', color='#333', linewidth=0.6, zorder=0)

    plt.suptitle('Heart Failure — Análise Individual do Paciente',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('resultado_inferencia_hf.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print('\n  Gráfico salvo em: resultado_inferencia_hf.png')


# ============================================================
# EXEMPLO DE INFERÊNCIA — PACIENTE HIPOTÉTICO
# ============================================================

print("\n" + "=" * 60)
print("  EXEMPLO DE INFERÊNCIA — PACIENTE HIPOTÉTICO")
print("=" * 60)

resultado, score, distribuicao = inferir_paciente(
    age=65,
    anaemia=1,                    # tem anemia
    creatinine_phosphokinase=250,
    diabetes=1,                   # tem diabetes
    ejection_fraction=30,         # baixa fração de ejeção (risco)
    high_blood_pressure=1,        # pressão alta
    platelets=200000,
    serum_creatinine=1.8,         # levemente elevado
    serum_sodium=135,
    sex=1,                        # masculino
    smoking=0,                    # não fuma
    time=60,                      # 60 dias de acompanhamento
)

exibir_resultado_inferencia(resultado, score, distribuicao)