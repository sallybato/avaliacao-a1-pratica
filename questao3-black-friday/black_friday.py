"""
Black Friday Sales — Sistema Inteligente de Classificação
==========================================================
Dataset : retail_black_friday_sales_100k.csv
          https://www.kaggle.com/datasets/noopurbhatt/retail-black-friday-sales-dataset
Targets : product_category, payment_method, age_group
Modelo  : Random Forest
Aluna   : Rebecca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    multilabel_confusion_matrix
)
from collections import Counter
from pickle import dump, load
from pprint import pprint
import warnings
warnings.filterwarnings('ignore')


# carregamento e pre-processamento

print("=" * 60)
print("  BLACK FRIDAY SALES — SISTEMA ")
print("=" * 60)

df = pd.read_csv('retail_black_friday_sales_100k.csv')

print(f"\nShape original: {df.shape}")
print(f"\nColunas: {list(df.columns)}")
print(f"\nPrimeiras linhas:")
print(df.head())
print(f"\nInformações:")
print(df.info())
print(f"\nValores nulos:\n{df.isnull().sum()}")

# análise dos targets

print(f"\n--- DISTRIBUIÇÃO DAS CLASSES ---")
print(f"\nProduct Category ({df['product_category'].nunique()} classes):")
print(df['product_category'].value_counts())
print(f"\nPayment Method ({df['payment_method'].nunique()} classes):")
print(df['payment_method'].value_counts())
print(f"\nAge Group ({df['age_group'].nunique()} classes):")
print(df['age_group'].value_counts())

# feature engineering

# Colunas descartadas: IDs e data (não são atributos preditivos)
df = df.drop(columns=['transaction_id', 'customer_id', 'product_id', 'purchase_date'])

# Encoding das colunas categóricas
le_gender   = LabelEncoder()
le_city     = LabelEncoder()
le_segment  = LabelEncoder()

df['gender']           = le_gender.fit_transform(df['gender'])
df['city']             = le_city.fit_transform(df['city'])
df['customer_segment'] = le_segment.fit_transform(df['customer_segment'])

print(f"\nClasses gender          : {le_gender.classes_}")
print(f"Classes city            : {le_city.classes_}")
print(f"Classes customer_segment: {le_segment.classes_}")

# Encoding dos targets
le_product  = LabelEncoder()
le_payment  = LabelEncoder()
le_age      = LabelEncoder()

df['product_category'] = le_product.fit_transform(df['product_category'])
df['payment_method']   = le_payment.fit_transform(df['payment_method'])
df['age_group']        = le_age.fit_transform(df['age_group'])

print(f"\nClasses product_category: {le_product.classes_}")
print(f"Classes payment_method  : {le_payment.classes_}")
print(f"Classes age_group       : {le_age.classes_}")

# features comuns usadas nos 3 classificadores

# is_weekend e is_black_friday já são binárias — não normalizar
# As demais features numéricas serão normalizadas

colunas_binarias  = ['is_weekend', 'is_black_friday']
colunas_continuas = ['original_price', 'discount_pct', 'final_price',
                     'quantity', 'purchase_amount', 'purchase_hour',
                     'gender', 'city', 'customer_segment']

print(f"\nColunas binárias (não normalizadas): {colunas_binarias}")
print(f"Colunas contínuas (normalizadas)   : {colunas_continuas}")

# função auxiliar, treinar e avaliar
def treinar_avaliar(X, y, nome_target, classes_labels):
    print(f"\n{'=' * 60}")
    print(f"  CLASSIFICADOR: {nome_target.upper()}")
    print(f"{'=' * 60}")

    pprint(dict(Counter(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='accuracy')

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    # Métricas globais
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    # Especificidade e sensibilidade por classe (via multilabel_confusion_matrix)
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    especificidades = []
    sensibilidades  = []
    for cm_bin in mcm:
        tn, fp, fn, tp = cm_bin.ravel()
        especificidades.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        sensibilidades.append(tp  / (tp + fn) if (tp + fn) > 0 else 0)

    esp_media = np.mean(especificidades)
    sen_media = np.mean(sensibilidades)

    print(f"\n  CV Accuracy (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Acurácia global       : {acc:.4f}")
    print(f"  F1-Score (weighted)   : {f1:.4f}")
    print(f"  Especificidade média  : {esp_media:.4f}")
    print(f"  Sensibilidade média   : {sen_media:.4f}")
    print(f"\n  Relatório completo:")
    print(classification_report(y_test, y_pred, target_names=classes_labels))

    cm = confusion_matrix(y_test, y_pred)

    return modelo, y_test, y_pred, y_proba, acc, f1, esp_media, sen_media, cm


# preparação das features

scaler = StandardScaler()

def preparar_X(df, target_col):
    """Retorna X sem o target e sem os outros targets."""
    outras_targets = ['product_category', 'payment_method', 'age_group']
    outras_targets = [c for c in outras_targets if c != target_col]
    X = df.drop(columns=outras_targets + [target_col])

    X_proc = X.copy()
    cols_cont = [c for c in colunas_continuas if c in X_proc.columns]
    X_proc[cols_cont] = scaler.fit_transform(X_proc[cols_cont])
    return X_proc


# CLASSIFICADOR 1: PRODUCT CATEGORY
X1 = preparar_X(df, 'product_category')
y1 = df['product_category']

modelo1, yt1, yp1, yproba1, acc1, f1_1, esp1, sen1, cm1 = treinar_avaliar(
    X1, y1, 'Categoria do Produto', le_product.classes_
)


# CLASSIFICADOR 2: PAYMENT METHOD
X2 = preparar_X(df, 'payment_method')
y2 = df['payment_method']

modelo2, yt2, yp2, yproba2, acc2, f1_2, esp2, sen2, cm2 = treinar_avaliar(
    X2, y2, 'Forma de Pagamento', le_payment.classes_
)


# CLASSIFICADOR 3: AGE GROUP
X3 = preparar_X(df, 'age_group')
y3 = df['age_group']

modelo3, yt3, yp3, yproba3, acc3, f1_3, esp3, sen3, cm3 = treinar_avaliar(
    X3, y3, 'Faixa Etária', le_age.classes_
)

# resumo geral 

print(f"\n{'=' * 60}")
print(f"  RESUMO GERAL DOS CLASSIFICADORES")
print(f"{'=' * 60}")
print(f"\n  {'Classificador':<25} {'Acurácia':>10} {'F1-Score':>10} {'Espec.':>10} {'Sensib.':>10}")
print(f"  {'-'*65}")
print(f"  {'Categoria do Produto':<25} {acc1:>10.4f} {f1_1:>10.4f} {esp1:>10.4f} {sen1:>10.4f}")
print(f"  {'Forma de Pagamento':<25} {acc2:>10.4f} {f1_2:>10.4f} {esp2:>10.4f} {sen2:>10.4f}")
print(f"  {'Faixa Etária':<25} {acc3:>10.4f} {f1_3:>10.4f} {esp3:>10.4f} {sen3:>10.4f}")

#salvando modelos e scaler

dump(modelo1, open('modelo_product_category.pkl', 'wb'))
dump(modelo2, open('modelo_payment_method.pkl',   'wb'))
dump(modelo3, open('modelo_age_group.pkl',        'wb'))
dump(scaler,  open('scaler_bf.pkl',               'wb'))
dump(le_product,  open('le_product.pkl',  'wb'))
dump(le_payment,  open('le_payment.pkl',  'wb'))
dump(le_age,      open('le_age.pkl',      'wb'))
dump(le_gender,   open('le_gender.pkl',   'wb'))
dump(le_city,     open('le_city.pkl',     'wb'))
dump(le_segment,  open('le_segment.pkl',  'wb'))

print(f"\n  Modelos e encoders salvos.")

# visualizações

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0f0f0f')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# Métricas comparativas
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#1a1a1a')
classificadores = ['Categoria\ndo Produto', 'Forma de\nPagamento', 'Faixa\nEtária']
accs = [acc1, acc2, acc3]
f1s  = [f1_1, f1_2, f1_3]
esps = [esp1, esp2, esp3]
sens = [sen1, sen2, sen3]
x    = np.arange(len(classificadores))
w    = 0.2
bars1 = ax1.bar(x - 1.5*w, accs, width=w, label='Acurácia',       color='#f4a7b9', alpha=0.85, zorder=3)
bars2 = ax1.bar(x - 0.5*w, f1s,  width=w, label='F1-Score',       color='#b5d5f5', alpha=0.85, zorder=3)
bars3 = ax1.bar(x + 0.5*w, esps, width=w, label='Especificidade', color='#c3e6cb', alpha=0.85, zorder=3)
bars4 = ax1.bar(x + 1.5*w, sens, width=w, label='Sensibilidade',  color='#ffe0b2', alpha=0.85, zorder=3)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, color='white')
ax1.set_xticks(x)
ax1.set_xticklabels(classificadores, color='white', fontsize=10)
ax1.set_ylim(0, 1.15)
ax1.set_title('Métricas por Classificador', color='white', fontsize=13, pad=12)
ax1.tick_params(colors='white')
ax1.spines[:].set_color('#333')
ax1.grid(axis='y', color='#333', linewidth=0.6, zorder=0)
ax1.legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#444')

# Matrizes de confusão
for i, (cm, classes, titulo, cor) in enumerate(zip(
    [cm1, cm2, cm3],
    [le_product.classes_, le_payment.classes_, le_age.classes_],
    ['Categoria do Produto', 'Forma de Pagamento', 'Faixa Etária'],
    ['#f4a7b9', '#b5d5f5', '#c3e6cb']
)):
    ax = fig.add_subplot(gs[1, i])
    ax.set_facecolor('#1a1a1a')
    ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, color='white', fontsize=6, rotation=45, ha='right')
    ax.set_yticklabels(classes, color='white', fontsize=6)
    ax.set_xlabel('Previsto', color='white', fontsize=8)
    ax.set_ylabel('Real',     color='white', fontsize=8)
    ax.set_title(f'Conf. Matrix — {titulo}', color=cor, fontsize=9, pad=8)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333')
    for r in range(len(classes)):
        for c in range(len(classes)):
            ax.text(c, r, cm[r, c], ha='center', va='center',
                    color='white', fontsize=6, fontweight='bold')

plt.suptitle('Black Friday Sales — Comparação dos Classificadores',
             color='white', fontsize=14, y=1.01)
plt.savefig('comparacao_classificadores_bf.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print('\nGráfico salvo em: comparacao_classificadores_bf.png')


# modulo de inferencia

def inferir_venda(
    gender, city, customer_segment,
    original_price, discount_pct, final_price,
    quantity, purchase_amount, purchase_hour,
    is_weekend, is_black_friday
):
    """
    Recebe os dados de uma venda e indica:
    - Categoria do produto mais provável
    - Forma de pagamento mais provável
    - Faixa etária mais provável do comprador
    Com grau de certeza para cada indicação.
    """
    # Carregar modelos e encoders
    m1  = load(open('modelo_product_category.pkl', 'rb'))
    m2  = load(open('modelo_payment_method.pkl',   'rb'))
    m3  = load(open('modelo_age_group.pkl',        'rb'))
    sc  = load(open('scaler_bf.pkl',               'rb'))
    lep = load(open('le_product.pkl',  'rb'))
    lem = load(open('le_payment.pkl',  'rb'))
    lea = load(open('le_age.pkl',      'rb'))
    leg = load(open('le_gender.pkl',   'rb'))
    lec = load(open('le_city.pkl',     'rb'))
    les = load(open('le_segment.pkl',  'rb'))

    # Encoding dos inputs categóricos
    gender_enc  = leg.transform([gender])[0]
    city_enc    = lec.transform([city])[0]
    segment_enc = les.transform([customer_segment])[0]

    # Monta o vetor base
    base = {
        'gender':           gender_enc,
        'city':             city_enc,
        'customer_segment': segment_enc,
        'original_price':   original_price,
        'discount_pct':     discount_pct,
        'final_price':      final_price,
        'quantity':         quantity,
        'purchase_amount':  purchase_amount,
        'purchase_hour':    purchase_hour,
        'is_weekend':       is_weekend,
        'is_black_friday':  is_black_friday,
    }

    def predizer(modelo, colunas_modelo, label_encoder):
        dados = pd.DataFrame([{k: base[k] for k in colunas_modelo}])
        cols_cont = [c for c in colunas_continuas if c in dados.columns]
        dados[cols_cont] = sc.transform(dados[cols_cont])
        classe   = modelo.predict(dados)[0]
        proba    = modelo.predict_proba(dados)[0]
        score    = max(proba)
        label    = label_encoder.inverse_transform([classe])[0]
        classes  = label_encoder.classes_
        return label, score, proba, classes

    cols1 = [c for c in X1.columns]
    cols2 = [c for c in X2.columns]
    cols3 = [c for c in X3.columns]

    prod,  score_prod,  proba_prod,  cls_prod  = predizer(m1, cols1, lep)
    pag,   score_pag,   proba_pag,   cls_pag   = predizer(m2, cols2, lem)
    idade, score_idade, proba_idade, cls_idade  = predizer(m3, cols3, lea)

    return (prod,  score_prod,  proba_prod,  cls_prod,
            pag,   score_pag,   proba_pag,   cls_pag,
            idade, score_idade, proba_idade, cls_idade)


def exibir_resultado_inferencia(
    prod,  score_prod,  proba_prod,  cls_prod,
    pag,   score_pag,   proba_pag,   cls_pag,
    idade, score_idade, proba_idade, cls_idade
):
    print("\n" + "=" * 55)
    print("  RESULTADO DO SISTEMA  ")
    print("=" * 55)
    print(f"\n  Categoria do Produto : {prod:<20} (certeza: {score_prod*100:.1f}%)")
    print(f"  Forma de Pagamento   : {pag:<20} (certeza: {score_pag*100:.1f}%)")
    print(f"  Faixa Etária         : {idade:<20} (certeza: {score_idade*100:.1f}%)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f0f')

    for ax, classes, proba, label, titulo, cor in zip(
        axes,
        [cls_prod,  cls_pag,   cls_idade],
        [proba_prod, proba_pag, proba_idade],
        [prod,       pag,       idade],
        ['Categoria do Produto', 'Forma de Pagamento', 'Faixa Etária'],
        ['#f4a7b9', '#b5d5f5', '#c3e6cb']
    ):
        ax.set_facecolor('#1a1a1a')
        cores_bar = [cor if c == label else '#555' for c in classes]
        bars = ax.bar(range(len(classes)), proba, color=cores_bar, alpha=0.85, zorder=3)
        for bar, val in zip(bars, proba):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val*100:.1f}%', ha='center', va='bottom',
                    color='white', fontsize=7, fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, color='white', fontsize=7, rotation=30, ha='right')
        ax.set_ylim(0, max(proba) * 1.3)
        ax.set_title(f'{titulo}\n→ {label}', color=cor, fontsize=10, pad=10)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333')
        ax.grid(axis='y', color='#333', linewidth=0.6, zorder=0)

    plt.suptitle('Black Friday — Sistema Inteligente de Classificação',
                 color='white', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('resultado_inferencia_bf.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print('\n  Gráfico salvo em: resultado_inferencia_bf.png')


# exemplo de inferencia

print("\n" + "=" * 60)
print("  EXEMPLO DE INFERÊNCIA — VENDA HIPOTÉTICA")
print("=" * 60)

resultados = inferir_venda(
    gender='Female',
    city='New York',
    customer_segment='VIP',
    original_price=299.99,
    discount_pct=20,
    final_price=239.99,
    quantity=1,
    purchase_amount=239.99,
    purchase_hour=14,
    is_weekend=0,
    is_black_friday=1,
)

exibir_resultado_inferencia(*resultados)