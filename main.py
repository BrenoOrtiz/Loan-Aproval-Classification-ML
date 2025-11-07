"""
PROJETO: CLASSIFICAÇÃO DE APROVAÇÃO DE EMPRÉSTIMOS
====================================================
Dataset: loan_data.csv
Target: loan_status (0 = Rejeitado, 1 = Aprovado)

Etapas:
1. Engenharia de Dados
2. Configuração e Treinamento de Modelos
3. Teste dos Modelos
4. Avaliação e Visualização dos Resultados
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend não-interativo para evitar problemas com Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PROJETO: CLASSIFICAÇÃO DE APROVAÇÃO DE EMPRÉSTIMOS")
print("="*80)

# ============================================================================
# ETAPA 1: ENGENHARIA DE DADOS
# ============================================================================
print("\n" + "="*80)
print("ETAPA 1: ENGENHARIA DE DADOS")
print("="*80)

# 1.1 - Carregar o dataset
print("\n[1.1] Carregando o dataset...")
df = pd.read_csv('loan_data.csv')
print(f"✓ Dataset carregado com sucesso!")
print(f"  - Shape: {df.shape}")
print(f"  - Registros: {df.shape[0]}")
print(f"  - Features: {df.shape[1]}")

# 1.2 - Análise exploratória inicial
print("\n[1.2] Análise Exploratória Inicial")
print("\nPrimeiras linhas do dataset:")
print(df.head())

print("\nInformações sobre o dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nDistribuição da variável target (loan_status):")
print(df['loan_status'].value_counts())
print("\nProporção:")
print(df['loan_status'].value_counts(normalize=True))

# 1.3 - Verificar valores ausentes
print("\n[1.3] Verificando Valores Ausentes")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("✓ Não há valores ausentes no dataset!")
else:
    print(f"⚠ Total de valores ausentes: {missing_values.sum()}")

# 1.4 - Tratamento de valores ausentes (se existirem)
print("\n[1.4] Tratamento de Valores Ausentes")
if df.isnull().sum().sum() > 0:
    # Para variáveis numéricas, preencher com a mediana
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  - {col}: preenchido com mediana")
    
    # Para variáveis categóricas, preencher com a moda
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  - {col}: preenchido com moda")
    print("✓ Valores ausentes tratados!")
else:
    print("✓ Nenhum tratamento necessário!")

# 1.5 - Detecção e tratamento de outliers
print("\n[1.5] Detecção e Tratamento de Outliers")
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('loan_status')  # Remover a variável target

outliers_count = {}
for col in numeric_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    outliers_count[col] = outliers

print("Número de outliers por feature:")
for col, count in outliers_count.items():
    if count > 0:
        print(f"  - {col}: {count} outliers ({count/len(df)*100:.2f}%)")

# Remover outliers extremos (opcional - comentado para manter mais dados)
# df = df[df['person_income'] < df['person_income'].quantile(0.99)]
# print("✓ Outliers extremos removidos!")

# 1.6 - Encoding de variáveis categóricas
print("\n[1.6] Encoding de Variáveis Categóricas")
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"Variáveis categóricas encontradas: {categorical_features}")

# Criar uma cópia do dataframe para encoding
df_encoded = df.copy()

# Label Encoding para variáveis ordinais
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  - {col}: {len(le.classes_)} categorias")

print("✓ Encoding concluído!")

# 1.7 - Criar novas features (Feature Engineering)
print("\n[1.7] Feature Engineering - Criando Novas Features")

# Razão de renda por idade
df_encoded['income_per_age'] = df_encoded['person_income'] / df_encoded['person_age']

# Razão de empréstimo por experiência de emprego
df_encoded['loan_per_exp'] = df_encoded['loan_amnt'] / (df_encoded['person_emp_exp'] + 1)

# Razão de taxa de juros por score de crédito
df_encoded['int_rate_per_credit'] = df_encoded['loan_int_rate'] / df_encoded['credit_score']

# Capacidade de pagamento (renda menos empréstimo)
df_encoded['payment_capacity'] = df_encoded['person_income'] - (df_encoded['loan_amnt'] * df_encoded['loan_percent_income'])

# Indicador de alto risco (taxa de juros > 15%)
df_encoded['high_risk_rate'] = (df_encoded['loan_int_rate'] > 15).astype(int)

# Score de crédito normalizado
df_encoded['credit_score_normalized'] = df_encoded['credit_score'] / 850  # Assumindo score máximo de 850

print("✓ Novas features criadas:")
print("  - income_per_age")
print("  - loan_per_exp")
print("  - int_rate_per_credit")
print("  - payment_capacity")
print("  - high_risk_rate")
print("  - credit_score_normalized")

# 1.8 - Separar features e target
print("\n[1.8] Separando Features e Target")
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

print(f"✓ Shape de X (features): {X.shape}")
print(f"✓ Shape de y (target): {y.shape}")

# 1.9 - Divisão em treino e teste
print("\n[1.9] Divisão em Conjuntos de Treino e Teste")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"✓ Distribuição no treino: {y_train.value_counts().to_dict()}")
print(f"✓ Distribuição no teste: {y_test.value_counts().to_dict()}")

# 1.10 - Normalização dos dados
print("\n[1.10] Normalização dos Dados (StandardScaler)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Dados normalizados (média=0, desvio=1)")
print(f"  - Média das features no treino (após escala): {X_train_scaled.mean():.6f}")
print(f"  - Desvio padrão das features no treino (após escala): {X_train_scaled.std():.6f}")

# ============================================================================
# ETAPA 2: CONFIGURAÇÃO E TREINAMENTO DE MODELOS
# ============================================================================
print("\n" + "="*80)
print("ETAPA 2: CONFIGURAÇÃO E TREINAMENTO DE MODELOS")
print("="*80)

# 2.1 - Definir modelos a serem testados
print("\n[2.1] Definindo Modelos de Machine Learning")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

print(f"✓ {len(models)} modelos configurados:")
for name in models.keys():
    print(f"  - {name}")

# 2.2 - Treinar modelos e avaliar com Cross-Validation
print("\n[2.2] Treinamento com Cross-Validation (5-fold)")
print("-" * 80)

cv_results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTreinando: {name}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    
    # Treinar modelo completo
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    print(f"  ✓ Acurácia média (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  ✓ Scores por fold: {[f'{s:.4f}' for s in cv_scores]}")

# 2.3 - Otimização de hiperparâmetros (GridSearch para Random Forest)
print("\n[2.3] Otimização de Hiperparâmetros (GridSearch - Random Forest)")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("Parâmetros testados:")
for param, values in param_grid.items():
    print(f"  - {param}: {values}")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

print("\nExecutando GridSearch... (pode levar alguns minutos)")
grid_search.fit(X_train_scaled, y_train)

print(f"\n✓ Melhores parâmetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  - {param}: {value}")
print(f"✓ Melhor acurácia (CV): {grid_search.best_score_:.4f}")

# Substituir Random Forest pelo modelo otimizado
trained_models['Random Forest (Optimized)'] = grid_search.best_estimator_

# ============================================================================
# ETAPA 3: TESTE DOS MODELOS
# ============================================================================
print("\n" + "="*80)
print("ETAPA 3: TESTE DOS MODELOS")
print("="*80)

# 3.1 - Fazer predições no conjunto de teste
print("\n[3.1] Realizando Predições no Conjunto de Teste")

predictions = {}
probabilities = {}

for name, model in trained_models.items():
    pred = model.predict(X_test_scaled)
    predictions[name] = pred
    
    # Probabilidades (se o modelo suportar)
    try:
        prob = model.predict_proba(X_test_scaled)[:, 1]
        probabilities[name] = prob
    except:
        probabilities[name] = None
    
    print(f"✓ {name}: {len(pred)} predições realizadas")

# ============================================================================
# ETAPA 4: AVALIAÇÃO E VISUALIZAÇÃO DOS RESULTADOS
# ============================================================================
print("\n" + "="*80)
print("ETAPA 4: AVALIAÇÃO E VISUALIZAÇÃO DOS RESULTADOS")
print("="*80)

# 4.1 - Calcular métricas de desempenho
print("\n[4.1] Métricas de Desempenho")
print("-" * 80)

results_df = []

for name, pred in predictions.items():
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    # AUC-ROC (se probabilidades estiverem disponíveis)
    if probabilities[name] is not None:
        auc = roc_auc_score(y_test, probabilities[name])
    else:
        auc = np.nan
    
    results_df.append({
        'Modelo': name,
        'Acurácia': accuracy,
        'Precisão': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    })
    
    print(f"\n{name}:")
    print(f"  - Acurácia:  {accuracy:.4f}")
    print(f"  - Precisão:  {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    if not np.isnan(auc):
        print(f"  - AUC-ROC:   {auc:.4f}")

results_df = pd.DataFrame(results_df)
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\n" + "="*80)
print("RESUMO DAS MÉTRICAS (Ordenado por F1-Score)")
print("="*80)
print(results_df.to_string(index=False))

# Salvar resultados
results_df.to_csv('resultados_loan_classification.csv', index=False)
print("\n✓ Resultados salvos em 'resultados_loan_classification.csv'")

# 4.2 - Visualização: Comparação de Modelos
print("\n[4.2] Gerando Visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparação de Modelos - Classificação de Aprovação de Empréstimos', 
             fontsize=16, fontweight='bold')

# Gráfico 1: Comparação de Acurácia
ax1 = axes[0, 0]
results_sorted = results_df.sort_values('Acurácia', ascending=True)
ax1.barh(results_sorted['Modelo'], results_sorted['Acurácia'], color='skyblue')
ax1.set_xlabel('Acurácia', fontweight='bold')
ax1.set_title('Comparação de Acurácia dos Modelos', fontweight='bold')
ax1.set_xlim([0, 1])
for i, v in enumerate(results_sorted['Acurácia']):
    ax1.text(v + 0.01, i, f'{v:.4f}', va='center')

# Gráfico 2: Comparação de F1-Score
ax2 = axes[0, 1]
results_sorted_f1 = results_df.sort_values('F1-Score', ascending=True)
ax2.barh(results_sorted_f1['Modelo'], results_sorted_f1['F1-Score'], color='lightcoral')
ax2.set_xlabel('F1-Score', fontweight='bold')
ax2.set_title('Comparação de F1-Score dos Modelos', fontweight='bold')
ax2.set_xlim([0, 1])
for i, v in enumerate(results_sorted_f1['F1-Score']):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center')

# Gráfico 3: Precisão vs Recall
ax3 = axes[1, 0]
for i, row in results_df.iterrows():
    ax3.scatter(row['Recall'], row['Precisão'], s=200, alpha=0.6, label=row['Modelo'])
ax3.set_xlabel('Recall', fontweight='bold')
ax3.set_ylabel('Precisão', fontweight='bold')
ax3.set_title('Precisão vs Recall', fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# Gráfico 4: Todas as métricas (radar plot simulado com barras agrupadas)
ax4 = axes[1, 1]
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.12

for i, (idx, row) in enumerate(results_df.head(5).iterrows()):
    values = [row['Acurácia'], row['Precisão'], row['Recall'], row['F1-Score']]
    ax4.bar(x + i*width, values, width, label=row['Modelo'], alpha=0.8)

ax4.set_xlabel('Métricas', fontweight='bold')
ax4.set_ylabel('Score', fontweight='bold')
ax4.set_title('Comparação de Todas as Métricas (Top 5 Modelos)', fontweight='bold')
ax4.set_xticks(x + width * 2)
ax4.set_xticklabels(metrics)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 'comparacao_modelos.png'")

# 4.3 - Matriz de Confusão para o melhor modelo
print("\n[4.3] Matriz de Confusão do Melhor Modelo")

best_model_name = results_df.iloc[0]['Modelo']
best_predictions = predictions[best_model_name]

print(f"\nMelhor Modelo: {best_model_name}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, best_predictions, 
                          target_names=['Rejeitado (0)', 'Aprovado (1)']))

# Plotar matriz de confusão
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Análise do Melhor Modelo: {best_model_name}', 
             fontsize=14, fontweight='bold')

# Matriz de confusão
cm = confusion_matrix(y_test, best_predictions)
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Rejeitado', 'Aprovado'],
            yticklabels=['Rejeitado', 'Aprovado'])
ax1.set_ylabel('Valor Real', fontweight='bold')
ax1.set_xlabel('Predição', fontweight='bold')
ax1.set_title('Matriz de Confusão', fontweight='bold')

# Curva ROC
if probabilities[best_model_name] is not None:
    fpr, tpr, _ = roc_curve(y_test, probabilities[best_model_name])
    auc = roc_auc_score(y_test, probabilities[best_model_name])
    
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Taxa de Falsos Positivos', fontweight='bold')
    ax2.set_ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
    ax2.set_title('Curva ROC', fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('melhor_modelo_analise.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 'melhor_modelo_analise.png'")

# 4.4 - Feature Importance (se o modelo suportar)
print("\n[4.4] Importância das Features")

best_model = trained_models[best_model_name]

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Features Mais Importantes:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plotar feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
    plt.xlabel('Importância', fontweight='bold')
    plt.title(f'Top 15 Features Mais Importantes - {best_model_name}', 
              fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico salvo: 'feature_importance.png'")
else:
    print(f"⚠ O modelo {best_model_name} não suporta feature_importances_")

# 4.5 - Análise de erros
print("\n[4.5] Análise de Erros")

errors_df = pd.DataFrame({
    'Real': y_test,
    'Predito': best_predictions
})
errors_df['Erro'] = errors_df['Real'] != errors_df['Predito']

total_errors = errors_df['Erro'].sum()
false_positives = ((errors_df['Real'] == 0) & (errors_df['Predito'] == 1)).sum()
false_negatives = ((errors_df['Real'] == 1) & (errors_df['Predito'] == 0)).sum()

print(f"\nTotal de erros: {total_errors} ({total_errors/len(y_test)*100:.2f}%)")
print(f"  - Falsos Positivos (aprovou quem deveria rejeitar): {false_positives}")
print(f"  - Falsos Negativos (rejeitou quem deveria aprovar): {false_negatives}")

# ============================================================================
# CONCLUSÃO
# ============================================================================
print("\n" + "="*80)
print("CONCLUSÃO DO PROJETO")
print("="*80)

print(f"\n✓ Modelo com melhor desempenho: {best_model_name}")
print(f"  - F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
print(f"  - Acurácia: {results_df.iloc[0]['Acurácia']:.4f}")
print(f"  - AUC-ROC: {results_df.iloc[0]['AUC-ROC']:.4f}")

print("\n✓ Arquivos gerados:")
print("  - resultados_loan_classification.csv")
print("  - comparacao_modelos.png")
print("  - melhor_modelo_analise.png")
print("  - feature_importance.png")

print("\n✓ Projeto concluído com sucesso!")
print("="*80)

# plt.show() removido - gráficos salvos em arquivos
