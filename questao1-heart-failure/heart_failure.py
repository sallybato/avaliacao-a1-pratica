"""
Heart Failure — Sistema de Classificação de Risco
Dataset : heart_failure_clinical_records_dataset.csv
          https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
Modelo  : Random Forest
Aluna   : Rebecca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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


#carregamento e pré-processamento

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

#análise de colunas binárias

# O dataset possui colunas binárias (0/1) que representam
# categorias e NÃO devem ser normalizadas pelo StandardScaler,
# pois já estão na escala correta e a normalização distorceria
# seu significado semântico (ex: anaemia = 0 ou 1).
#
# Colunas binárias identificadas:
#   - anaemia             (0 = não, 1 = sim)
#   - diabetes            (0 = não, 1 = sim)
#   - high_blood_pressure (0 = não, 1 = sim)
#   - sex                 (0 = feminino, 1 = masculino)
#   - smoking             (0 = não, 1 = sim)
#
# Estratégia foi escalonar APENAS as colunas contínuas.

colunas_binarias  = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
colunas_continuas = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                     'platelets', 'serum_creatinine', 'serum_sodium', 'time']

print(f"\n--- COLUNAS BINÁRIAS (não serão normalizadas) ---")
print(colunas_binarias)
print(f"\n--- COLUNAS CONTÍNUAS (serão normalizadas) ---")
print(colunas_continuas)

# Verificar valores nulos
print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")

# Distribuição do target
print(f"\nDistribuição da classe (DEATH_EVENT):")
print(df['DEATH_EVENT'].value_counts())
print(df['DEATH_EVENT'].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))

#separação de atributos e target

X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

#pré-processamento e escalonamento seletivo

scaler = StandardScaler()
X_processado = X.copy()
X_processado[colunas_continuas] = scaler.fit_transform(X[colunas_continuas])

print(f"\n--- DADOS APÓS PRÉ-PROCESSAMENTO ---")
print(X_processado.describe().round(3))

# divisao de treino/teste

print(f"\n### FREQUÊNCIA DAS CLASSES ###")
pprint(dict(Counter(y)))

X_train, X_test, y_train, y_test = train_test_split(
    X_processado, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTreino : {X_train.shape[0]} amostras")
print(f"Teste  : {X_test.shape[0]} amostras")

# definição e treinamento do modelo

# Metaestimador escolhido: Random Forest
# Justificativa: algoritmo ensemble baseado em múltiplas árvores
# de decisão, robusto para datasets pequenos (299 pacientes),
# lida bem com o mix de variáveis binárias e contínuas sem
# distorcer o significado semântico das binárias, e o parâmetro
# class_weight='balanced' compensa o desbalanceamento entre
# pacientes que sobreviveram e que foram a óbito.

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print(f"  Treinando: Random Forest")
print("=" * 60)

# Validação cruzada
cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')

# Treino final
modelo.fit(X_train, y_train)
y_pred  = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()
especificidade = tn / (tn + fp)
sensibilidade  = tp / (tp + fn)

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
print(classification_report(y_test, y_pred, target_names=['Sobreviveu', 'Faleceu']))

#salvando modelo e scaler

dump(modelo,  open('melhor_modelo_hf.pkl', 'wb'))
dump(scaler,  open('scaler_hf.pkl', 'wb'))

print(f"  Modelo salvo em: melhor_modelo_hf.pkl")
print(f"  Scaler salvo em: scaler_hf.pkl")

# visualizações

fig = plt.figure(figsize=(14, 5))
fig.patch.set_facecolor('#0f0f0f')
gs  = gridspec.GridSpec(1, 3, figure=fig, hspace=0.4, wspace=0.4)

# Métricas em barras
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1a1a')
metricas = ['Acurácia', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
valores  = [acc, prec, rec, f1, auc]
bars = ax1.bar(metricas, valores, color='#f4a7b9', alpha=0.85, zorder=3)
for bar, val in zip(bars, valores):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='white')
ax1.set_ylim(0, 1.15)
ax1.set_title('Métricas — Random Forest', color='white', fontsize=11, pad=10)
ax1.tick_params(colors='white', labelsize=7)
ax1.spines[:].set_color('#333')
ax1.grid(axis='y', color='#333', linewidth=0.6, zorder=0)

# Matriz de confusão
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1a1a1a')
ax2.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Sobreviveu', 'Faleceu'], color='white', fontsize=8)
ax2.set_yticklabels(['Sobreviveu', 'Faleceu'], color='white', fontsize=8)
ax2.set_xlabel('Previsto', color='white', fontsize=9)
ax2.set_ylabel('Real',     color='white', fontsize=9)
ax2.set_title('Matriz de Confusão', color='#f4a7b9', fontsize=11, pad=10)
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#333')
for r in range(2):
    for c in range(2):
        ax2.text(c, r, cm[r, c], ha='center', va='center',
                 color='white', fontsize=14, fontweight='bold')

# Curva ROC
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor('#1a1a1a')
RocCurveDisplay.from_predictions(
    y_test, y_proba,
    name=f"Random Forest (AUC={auc:.3f})",
    ax=ax3, color='#f4a7b9'
)
ax3.plot([0, 1], [0, 1], 'w--', lw=0.8, label='Baseline')
ax3.set_title('Curva ROC', color='white', fontsize=11, pad=10)
ax3.tick_params(colors='white')
ax3.spines[:].set_color('#444')
ax3.xaxis.label.set_color('white')
ax3.yaxis.label.set_color('white')
ax3.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#555', fontsize=8)
ax3.grid(color='#333', linewidth=0.5)

plt.suptitle('Heart Failure — Random Forest', color='white', fontsize=13, y=1.02)
plt.savefig('resultado_modelo_hf.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print('\nGráfico salvo em: resultado_modelo_hf.png')


# módulo de inferencia

def inferir_paciente(
    age, anaemia, creatinine_phosphokinase, diabetes,
    ejection_fraction, high_blood_pressure, platelets,
    serum_creatinine, serum_sodium, sex, smoking, time,
    modelo_path='melhor_modelo_hf.pkl',
    scaler_path='scaler_hf.pkl'
):
    """
    Recebe os dados de um paciente desconhecido e indica
    a qual grupo ele pertence (Sobreviveu / Risco de Óbito).
    """
    modelo_inf = load(open(modelo_path, 'rb'))
    scaler_inf = load(open(scaler_path, 'rb'))

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

    dados[colunas_continuas] = scaler_inf.transform(dados[colunas_continuas])

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


# apenas um exemplo de inferencia

print("\n" + "=" * 60)
print("  EXEMPLO DE INFERÊNCIA — PACIENTE HIPOTÉTICO")
print("=" * 60)

resultado, score, distribuicao = inferir_paciente(
    age=65,
    anaemia=1,
    creatinine_phosphokinase=250,
    diabetes=1,
    ejection_fraction=30,
    high_blood_pressure=1,
    platelets=200000,
    serum_creatinine=1.8,
    serum_sodium=135,
    sex=1,
    smoking=0,
    time=60,
)

exibir_resultado_inferencia(resultado, score, distribuicao)