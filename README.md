# Detecção de Anomalias em Transações (Fraudes de Cartão de Crédito) 🕵️‍♂️💳

Este é o projeto final do desafio de **Detecção de Anomalias em Transações em Python** da plataforma DIO.

## 🎯 Objetivo
O principal objetivo deste projeto é criar um modelo de Machine Learning capaz de identificar transações fraudulentas de cartões de crédito. O grande desafio aqui é lidar com **dados altamente desbalanceados**, onde as fraudes representam menos de 0,2% de todas as transações.

## 🛠️ Tecnologias e Técnicas Utilizadas
- **Python:** Linguagem principal do projeto.
- **Pandas & NumPy:** Para manipulação e análise dos dados.
- **Scikit-Learn:** Para criar o modelo de Inteligência Artificial e separar os dados.
- **Imbalanced-Learn (imblearn):** Para resolver o problema de desbalanceamento usando a técnica de *Undersampling*.
- **Matplotlib & Seaborn:** Para a criação dos gráficos visuais (Matriz de Confusão).

## 🧠 Metodologia
1. **Análise Inicial:** Foi verificado que o conjunto original contava com cerca de 284.000 transações reais, sendo apenas 492 fraudes (0,17%).
2. **Preparação e Padronização:** As colunas de "Tempo" e "Valor" da transação foram padronizadas (`StandardScaler`) para que os valores financeiros altos não confundissem o modelo.
3. **Divisão:** 80% dos dados foram separados para treinamento e 20% guardados para o teste final.
4. **Undersampling:** Para evitar a "Paradoxo da Acurácia", os dados de treinamento foram balanceados, sorteando 394 transações normais para bater igual com as 394 fraudes do treino (50/50).
5. **Treinamento e Avaliação:** Foi utilizado o modelo de `Regressão Logística`. Ao invés da Acurácia, o foco principal foi na métrica de **Recall** (Revocação).

## 📊 Resultados Alcançados
No teste cego (prova final) contendo 56.962 transações, nosso modelo obteve um **Recall de 92%**!
Isso significa que das **98 fraudes** reais que estavam escondidas no teste, a IA conseguiu detectar **90** de imediato, deixando passar apenas 8. 

Isso se provou um excelente resultado base para a prevenção de perdas financeiras.
