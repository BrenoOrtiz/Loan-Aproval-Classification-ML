# 🏦 Classificação de Aprovação de Empréstimos

## 📋 Descrição do Projeto

Este projeto implementa um sistema de **Machine Learning** para classificar solicitações de empréstimo como **aprovadas** ou **rejeitadas**, utilizando técnicas de engenharia de dados e múltiplos algoritmos de classificação.

**Dataset:** `loan_data.csv`  
**Variável Target:** `loan_status`

-   **0**: Empréstimo Rejeitado
-   **1**: Empréstimo Aprovado

---

## 🎯 Objetivos

1. ✅ Aplicar técnicas de **Engenharia de Dados**
2. ✅ Configurar e treinar **modelos de Machine Learning**
3. ✅ Testar e avaliar os modelos
4. ✅ Visualizar e comparar os resultados

---

## 📊 Dataset

O dataset contém **45.002 registros** com as seguintes features:

### Features Originais:

-   `person_age`: Idade da pessoa
-   `person_gender`: Gênero (female/male)
-   `person_education`: Nível educacional (High School, Bachelor, Master, Associate)
-   `person_income`: Renda anual
-   `person_emp_exp`: Anos de experiência profissional
-   `person_home_ownership`: Tipo de moradia (RENT, OWN, MORTGAGE, OTHER)
-   `loan_amnt`: Valor do empréstimo solicitado
-   `loan_intent`: Propósito do empréstimo (PERSONAL, EDUCATION, MEDICAL, VENTURE, etc.)
-   `loan_int_rate`: Taxa de juros do empréstimo
-   `loan_percent_income`: Percentual da renda comprometido
-   `cb_person_cred_hist_length`: Histórico de crédito (anos)
-   `credit_score`: Score de crédito
-   `previous_loan_defaults_on_file`: Inadimplência anterior (Yes/No)
-   `loan_status`: **Target** - Status de aprovação (0/1)

---

## 🔧 Engenharia de Dados

### 1. Análise Exploratória

-   Verificação da estrutura do dataset
-   Análise estatística descritiva
-   Distribuição da variável target
-   Identificação de valores ausentes

### 2. Pré-processamento

-   **Tratamento de valores ausentes**:

    -   Variáveis numéricas: preenchidas com a mediana
    -   Variáveis categóricas: preenchidas com a moda

-   **Detecção de Outliers**:
    -   Método IQR (Interquartile Range)
    -   Identificação de valores extremos

### 3. Encoding de Variáveis Categóricas

-   **Label Encoding** aplicado a:
    -   `person_gender`
    -   `person_education`
    -   `person_home_ownership`
    -   `loan_intent`
    -   `previous_loan_defaults_on_file`

### 4. Feature Engineering (Novas Features Criadas)

Foram criadas **6 novas features** para melhorar o poder preditivo:

1. **`income_per_age`**: Renda dividida pela idade

    - Indica capacidade financeira relativa à idade

2. **`loan_per_exp`**: Valor do empréstimo dividido pela experiência profissional

    - Avalia adequação do empréstimo à experiência

3. **`int_rate_per_credit`**: Taxa de juros dividida pelo score de crédito

    - Relação entre risco e taxa cobrada

4. **`payment_capacity`**: Capacidade de pagamento

    - Renda menos (valor do empréstimo × percentual da renda)

5. **`high_risk_rate`**: Indicador de alto risco

    - 1 se taxa de juros > 15%, 0 caso contrário

6. **`credit_score_normalized`**: Score de crédito normalizado
    - Score dividido por 850 (valor máximo assumido)

### 5. Normalização

-   **StandardScaler**: Normalização de todas as features (média=0, desvio=1)
-   Melhora convergência e desempenho dos modelos

### 6. Divisão dos Dados

-   **Treino**: 80% (36.001 amostras)
-   **Teste**: 20% (9.001 amostras)
-   **Estratificação**: Mantém proporção da variável target

---

## 🤖 Modelos de Machine Learning

### Modelos Treinados

Foram testados **6 algoritmos** diferentes:

1. **Logistic Regression** (Regressão Logística)

    - Modelo linear probabilístico
    - Baseline simples e interpretável

2. **Decision Tree** (Árvore de Decisão)

    - Modelo baseado em regras
    - Alta interpretabilidade

3. **Random Forest** (Floresta Aleatória)

    - Ensemble de árvores de decisão
    - Robusto e com boa generalização

4. **Gradient Boosting** (Boosting Gradiente)

    - Ensemble sequencial
    - Alto poder preditivo

5. **K-Nearest Neighbors** (KNN)

    - Classificação por proximidade
    - Não-paramétrico

6. **Support Vector Machine** (SVM)
    - Classificação por hiperplano
    - Eficaz em espaços de alta dimensão

### Validação Cruzada

-   **5-fold Cross-Validation** aplicado em todos os modelos
-   Garante robustez e evita overfitting

### Otimização de Hiperparâmetros

**GridSearchCV** aplicado ao Random Forest:

Parâmetros testados:

-   `n_estimators`: [100, 200]
-   `max_depth`: [10, 20, None]
-   `min_samples_split`: [2, 5]
-   `min_samples_leaf`: [1, 2]

**Total de combinações**: 24 configurações testadas

---

## 📈 Métricas de Avaliação

Todas as métricas foram calculadas no conjunto de teste:

### Métricas Utilizadas:

1. **Acurácia** (Accuracy)

    - Proporção de predições corretas
    - Fórmula: (VP + VN) / Total

2. **Precisão** (Precision)

    - Proporção de positivos corretamente identificados
    - Fórmula: VP / (VP + FP)
    - Responde: "Dos empréstimos aprovados, quantos deveriam ser?"

3. **Recall** (Sensibilidade)

    - Proporção de positivos encontrados
    - Fórmula: VP / (VP + FN)
    - Responde: "Dos que deveriam ser aprovados, quantos foram?"

4. **F1-Score**

    - Média harmônica entre Precisão e Recall
    - Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
    - Balanceia ambas as métricas

5. **AUC-ROC** (Area Under the Curve)
    - Área sob a curva ROC
    - Mede capacidade de discriminação do modelo

### Legenda:

-   **VP** (Verdadeiro Positivo): Aprovado corretamente
-   **VN** (Verdadeiro Negativo): Rejeitado corretamente
-   **FP** (Falso Positivo): Aprovou quem deveria rejeitar
-   **FN** (Falso Negativo): Rejeitou quem deveria aprovar

---

## 📊 Resultados

### Comparação de Modelos

Os resultados estão ordenados por **F1-Score** (métrica mais equilibrada):

| Posição | Modelo                    | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
| ------- | ------------------------- | -------- | -------- | ------ | -------- | ------- |
| 🥇 1º   | Random Forest (Optimized) | ~0.93    | ~0.92    | ~0.94  | ~0.93    | ~0.97   |
| 🥈 2º   | Random Forest             | ~0.92    | ~0.91    | ~0.93  | ~0.92    | ~0.97   |
| 🥉 3º   | Gradient Boosting         | ~0.92    | ~0.91    | ~0.93  | ~0.92    | ~0.97   |
| 4º      | Logistic Regression       | ~0.89    | ~0.88    | ~0.90  | ~0.89    | ~0.95   |
| 5º      | SVM                       | ~0.88    | ~0.87    | ~0.89  | ~0.88    | ~0.94   |
| 6º      | Decision Tree             | ~0.86    | ~0.85    | ~0.87  | ~0.86    | ~0.86   |
| 7º      | KNN                       | ~0.85    | ~0.84    | ~0.86  | ~0.85    | ~0.90   |

**Nota**: Os valores são aproximados e podem variar ligeiramente a cada execução devido à aleatoriedade dos algoritmos.

### 🏆 Melhor Modelo: Random Forest (Otimizado)

**Por que o Random Forest venceu?**

-   ✅ Excelente balanceamento entre Precisão e Recall
-   ✅ Alta capacidade de generalização
-   ✅ Robusto a outliers e dados ruidosos
-   ✅ Captura relações não-lineares complexas
-   ✅ Reduz overfitting através de ensemble

### Análise de Erros do Melhor Modelo

-   **Taxa de Erro**: ~7%
-   **Falsos Positivos**: Aprovou empréstimos de alto risco (~3-4%)
-   **Falsos Negativos**: Rejeitou bons pagadores (~3-4%)

---

## 🎨 Visualizações Geradas

### 1. `comparacao_modelos.png`

Contém 4 gráficos:

-   **Gráfico 1**: Comparação de Acurácia (barras horizontais)
-   **Gráfico 2**: Comparação de F1-Score (barras horizontais)
-   **Gráfico 3**: Precisão vs Recall (scatter plot)
-   **Gráfico 4**: Todas as métricas dos top 5 modelos (barras agrupadas)

### 2. `melhor_modelo_analise.png`

Análise detalhada do melhor modelo:

-   **Matriz de Confusão**: Visualização de acertos e erros
-   **Curva ROC**: Taxa de Verdadeiros Positivos vs Falsos Positivos

### 3. `feature_importance.png`

-   **Top 15 Features Mais Importantes**
-   Mostra quais variáveis mais influenciam a decisão do modelo
-   Gráfico de barras horizontais com importância relativa de cada feature
-   Baseado no modelo Random Forest Otimizado (melhor desempenho)

### 4. `resultados_loan_classification.csv`

-   Tabela com todas as métricas de todos os modelos
-   Formato CSV para análise posterior

---

## 🔍 Features Mais Importantes

As features que mais influenciam a aprovação de empréstimos são (baseado no Random Forest Otimizado):

1. 🥇 **`previous_loan_defaults_on_file`**: Histórico de inadimplência anterior

    - **Importância**: ~0.24 (24%)
    - O fator mais decisivo - inadimplência prévia reduz drasticamente as chances

2. 🥈 **`loan_percent_income`**: Percentual da renda comprometido com o empréstimo

    - **Importância**: ~0.14 (14%)
    - Quanto maior o comprometimento da renda, maior o risco

3. 🥉 **`loan_int_rate`**: Taxa de juros do empréstimo

    - **Importância**: ~0.09 (9%)
    - Taxas altas indicam perfis de maior risco

4. **`int_rate_per_credit`**: Taxa de juros dividida pelo score de crédito _(feature criada)_

    - **Importância**: ~0.07 (7%)
    - Relação entre risco percebido e score

5. **`payment_capacity`**: Capacidade de pagamento _(feature criada)_

    - **Importância**: ~0.07 (7%)
    - Renda disponível após comprometimento com empréstimo

6. **`person_income`**: Renda anual da pessoa

    - **Importância**: ~0.07 (7%)
    - Maior renda aumenta capacidade de pagamento

7. **`person_home_ownership`**: Tipo de moradia (própria, alugada, financiada)

    - **Importância**: ~0.06 (6%)
    - Estabilidade patrimonial

8. **`income_per_age`**: Renda dividida pela idade _(feature criada)_

    - **Importância**: ~0.05 (5%)
    - Capacidade financeira relativa à idade

9. **`loan_amnt`**: Valor do empréstimo solicitado

    - **Importância**: ~0.04 (4%)
    - Valores muito altos aumentam o risco

10. **`high_risk_rate`**: Indicador de taxa de juros alta (>15%) _(feature criada)_

    - **Importância**: ~0.03 (3%)
    - Sinalizador binário de risco elevado

11. **`credit_score_normalized`**: Score de crédito normalizado _(feature criada)_

    - **Importância**: ~0.03 (3%)
    - Score padronizado entre 0 e 1

12. **`credit_score`**: Score de crédito original

    - **Importância**: ~0.03 (3%)
    - Medida tradicional de confiabilidade financeira

13. **`loan_intent`**: Propósito do empréstimo

    - **Importância**: ~0.03 (3%)
    - Tipo de uso influencia aprovação

14. **`loan_per_exp`**: Valor do empréstimo pela experiência profissional _(feature criada)_

    - **Importância**: ~0.03 (3%)
    - Adequação do valor à experiência

15. **`person_age`**: Idade da pessoa
    - **Importância**: ~0.02 (2%)
    - Fator demográfico complementar

### 📊 Insights sobre Feature Importance:

**🎯 Descobertas Principais:**

1. **Inadimplência Anterior Domina**: Com 24% de importância, o histórico de inadimplência é DISPARADO o fator mais importante, sendo quase 2x mais relevante que o segundo colocado.

2. **Features Criadas São Valiosas**: Das 15 features mais importantes, **5 são features criadas** através de Feature Engineering:

    - `int_rate_per_credit` (4º lugar)
    - `payment_capacity` (5º lugar)
    - `income_per_age` (8º lugar)
    - `high_risk_rate` (10º lugar)
    - `credit_score_normalized` (11º lugar)
    - `loan_per_exp` (14º lugar)

3. **Comprometimento da Renda**: `loan_percent_income` (14%) é o segundo fator mais importante, mostrando que o percentual da renda comprometido é crítico.

4. **Score de Crédito Não É Tudo**: Embora importante, o `credit_score` original aparece apenas em 12º lugar (3%), sendo menos relevante que features derivadas.

5. **Top 3 Representa 47%**: As três primeiras features sozinhas representam quase metade da importância total do modelo.

---

## 🚀 Como Executar

### Pré-requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execução

```cmd
python main.py
```

### Saída Esperada

O script irá:

1. ✅ Carregar e processar os dados
2. ✅ Treinar 6 modelos diferentes
3. ✅ Otimizar hiperparâmetros
4. ✅ Avaliar todos os modelos
5. ✅ Gerar 3 arquivos de visualização (PNG)
6. ✅ Salvar resultados em CSV
7. ✅ Exibir relatório detalhado no terminal

---

## 📁 Estrutura de Arquivos

```
TDE/
├── loan_data.csv                          # Dataset original
├── main.py                                 # Código principal
├── README.md                               # Este arquivo
├── resultados_loan_classification.csv     # Resultados (gerado)
├── comparacao_modelos.png                 # Gráficos comparativos (gerado)
├── melhor_modelo_analise.png              # Análise do melhor modelo (gerado)
└── feature_importance.png                 # Importância das features (gerado)
```

---

## 💡 Insights e Conclusões

### Principais Descobertas:

1. **Inadimplência Anterior é o Fator Crítico** 🚨

    - Representa **24%** da importância total do modelo
    - É o preditor MAIS importante, quase 2x mais relevante que o segundo colocado
    - Inadimplência prévia reduz drasticamente as chances de aprovação
    - **Ação recomendada**: Criar políticas rigorosas para perfis com histórico negativo

2. **Comprometimento de Renda é Decisivo** 💰

    - `loan_percent_income` é o **2º fator mais importante (14%)**
    - Quanto maior o percentual da renda comprometido, maior o risco
    - **Ação recomendada**: Estabelecer limites máximos de comprometimento (ex: 30-40% da renda)

3. **Taxa de Juros como Indicador de Risco** 📈

    - `loan_int_rate` é o **3º fator (9%)**
    - Taxas altas (>15%) indicam perfis de maior risco
    - Correlacionada com probabilidade de rejeição
    - **Ação recomendada**: Usar taxa de juros como screening inicial

4. **Feature Engineering Teve Grande Impacto** ⚙️

    - **6 das 15 features mais importantes foram criadas** manualmente
    - `int_rate_per_credit`, `payment_capacity`, `income_per_age`, etc.
    - Demonstra que conhecimento do domínio + criatividade > dados brutos
    - **Conclusão**: Investir em Feature Engineering vale muito a pena!

5. **Score de Crédito Não É o Rei** 👑

    - Ao contrário do esperado, `credit_score` aparece apenas em **12º lugar (3%)**
    - Features derivadas como `int_rate_per_credit` são mais importantes
    - **Insight**: O contexto do score importa mais que o valor absoluto

6. **Top 3 Domina o Modelo** 🏆
    - As 3 primeiras features representam **47%** da importância total
    - Foco em: inadimplência anterior, comprometimento de renda e taxa de juros
    - **Ação recomendada**: Priorizar a qualidade e validação dessas 3 features

### Recomendações Práticas:

✅ **Para Instituições Financeiras**:

-   ⚠️ **Prioridade máxima**: Verificar histórico de inadimplência
-   📊 Estabelecer limite máximo de comprometimento de renda (ex: 35%)
-   🔍 Usar taxa de juros como indicador de risco inicial
-   💡 Investir em Feature Engineering para criar métricas compostas
-   📈 Considerar `payment_capacity` além de apenas renda bruta
-   🎯 Não confiar apenas no score de crédito - analisar contexto

✅ **Para Solicitantes**:

-   ✨ **Mais importante**: Manter histórico limpo (sem inadimplências)
-   💵 Solicitar valores que não comprometam mais de 30-35% da renda
-   📉 Buscar taxas de juros competitivas (abaixo de 15%)
-   💼 Demonstrar estabilidade (moradia própria, experiência profissional)
-   🏠 Tipo de moradia influencia (própria > financiada > alugada)

✅ **Para o Modelo em Produção**:

-   ✅ Random Forest otimizado é a melhor escolha (F1-Score ~0.93)
-   ✅ Modelo equilibrado entre precisão e recall
-   ✅ 6 features criadas melhoraram significativamente o desempenho
-   ⚠️ Monitorar continuamente as top 3 features
-   🔄 Retreinar periodicamente com novos dados
-   📊 Implementar sistema de explicabilidade (SHAP values)

---

## 🛠️ Tecnologias Utilizadas

-   **Python 3.13**
-   **Pandas**: Manipulação de dados
-   **NumPy**: Computação numérica
-   **Scikit-learn**: Modelos de ML
-   **Matplotlib**: Visualizações
-   **Seaborn**: Gráficos estatísticos

---

## 📚 Metodologia

### Pipeline Completo:

```
1. Dados Brutos (loan_data.csv)
         ↓
2. Análise Exploratória
         ↓
3. Limpeza e Pré-processamento
         ↓
4. Feature Engineering
         ↓
5. Encoding e Normalização
         ↓
6. Divisão Treino/Teste (80/20)
         ↓
7. Treinamento de 6 Modelos
         ↓
8. Cross-Validation (5-fold)
         ↓
9. Otimização de Hiperparâmetros
         ↓
10. Avaliação no Conjunto de Teste
         ↓
11. Visualização e Análise de Resultados
```

---

## 📊 Estatísticas do Dataset

-   **Total de Registros**: 45.002
-   **Features Originais**: 14
-   **Features Após Engineering**: 20
-   **Proporção da Target**:
    -   Aprovados (1): ~22%
    -   Rejeitados (0): ~77%

## 📝 Licença

Este projeto é para fins educacionais.

---
