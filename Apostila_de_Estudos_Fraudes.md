# Apostila de Estudos: Detecção de Fraudes em Cartão de Crédito 🕵️‍♂️💳

Se você achou esse projeto difícil, não se sinta mal! Ele é famoso por ser o "divisor de águas" no aprendizado de Inteligência Artificial. Diferente de projetos onde tudo dá certo de primeira, esse projeto esconde "armadilhas matemáticas".

Esta apostila foi feita para você ler com calma depois, entender o "porquê" de cada código e poder refazer tudo sozinho.

---

## 🛑 O Ponto Cego Número 1: O Paradoxo da Acurácia

A maior armadilha desse projeto está na **quantidade de dados**. Nós tínhamos:
- **284.315** transações normais.
- **492** fraudes.

As fraudes representavam apenas **0,17%**. 

Se você criar um modelo "burro" que foi programado para dizer apenas *"nenhuma transação é fraude"*, ele vai acertar 99,83% das vezes. O chefe veria a **Acurácia (taxa de acerto geral)** de 99,83% e ficaria feliz. 
O Ponto Cego: **Todas as 492 fraudes passariam pelo banco**, pois o modelo não sabe o que é uma fraude.

**Lição:** Em problemas desbalanceados (fraudes, doenças raras), **NUNCA** olhe para a Acurácia. Olhe para o **Recall**.

---

## 🛠️ Passo a Passo Explicado

### Passo 1: Importar Ferramentas e Dados
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# df significa DataFrame (sua tabela de dados)
df = pd.read_csv('creditcard.csv')
```
**O que isso faz?** Transforma o arquivo de 150MB do Excel em uma tabela na memória do Python para podermos trabalhar.

### Passo 2: Padronização (O Ponto Cego do Dinheiro)
Imagine que um cartão comprou um chiclete de R$ 2,00 e o outro comprou um carro de R$ 90.000,00. Números muito grandes confundem a matemática da IA.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
```
**O que isso faz?** O `StandardScaler` pega todos os valores de dinheiro (`Amount`) e tempo (`Time`) e "espreme" eles entre números como -1 e 1. A proporção continua a mesma, mas a IA não se assusta mais com valores gigantes.

### Passo 3: Dividir para Conquistar (Treino e Teste)
Você não pode dar a prova final para o aluno antes dele estudar.

```python
from sklearn.model_selection import train_test_split

# X = Todo o histórico do cartão (Tempo, Valor, Local...)
X = df.drop('Class', axis=1)

# y = A resposta final (0 = Normal, 1 = Fraude)
y = df['Class']

# Separando 80% pra Treino e 20% pra Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
**O que isso faz?** O `stratify=y` garante que a mesma proporção de fraudes que existe no mundo real (0,17%) seja mandada tanto para o Treino quanto para o Teste. 

### Passo 4: Undersampling (A Solução do Ponto Cego)
Lembra que o modelo poderia viciar em dizer "é tudo normal"? Aqui está o remédio.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_balanceado, y_train_balanceado = rus.fit_resample(X_train, y_train)
```
**O que isso faz?** Ele joga fora centenas de milhares de transações normais, até que sobrem exatamente 394 normais para bater de frente com as 394 fraudes do treino. Agora, a IA vai estudar em um ambiente **50% Fraude / 50% Normal**. Ela é obrigada a aprender as diferenças!

### Passo 5: Criando e Treinando a IA
```python
from sklearn.linear_model import LogisticRegression

# 1. Cria o modelo matemático
modelo = LogisticRegression()

# 2. .fit() significa "Estudar/Treinar"
modelo.fit(X_train_balanceado, y_train_balanceado)

# 3. .predict() significa "Faça a prova agora"
previsoes = modelo.predict(X_test)
```
**O que isso faz?** O `LogisticRegression` traça uma linha separando os dados. Tudo que cai de um lado da linha ele bloqueia como fraude, tudo do outro lado ele libera. O `fit` é onde essa linha é desenhada matematicamente.

### Passo 6: Lendo o "Boletim" da IA
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, previsoes))
```
**O que isso faz?** Ele imprime 3 métricas principais.
1. **Precision (Precisão):** De todas as vezes que a IA gritou *"É FRAUDE!"*, quantas ela realmente estava certa? (Aqui o número será baixo, uns 4%, o que significa que ela acusou muito inocente).
2. **Recall (Revocação):** De todas as fraudes *verdadeiras* que estavam escondidas, quantas a IA conseguiu achar? (Nosso número foi **92%**! Excelente!).
3. **F1-Score:** A média entre Precision e Recall.

**Por que o Banco ama o nosso modelo, mesmo a Precisão sendo baixa?**
No mundo real, se a IA desconfiar de uma transação normal (Falso Positivo), o banco bloqueia o cartão temporariamente e manda um SMS: *"Foi você que tentou comprar 5 mil na loja X?"*. O cliente diz "Sim", o cartão libera. Dá um leve aborrecimento, mas o dinheiro está a salvo.
Porém, se a IA não ver uma fraude real (Falso Negativo), o golpista rouba os 5 mil e o banco tem que pagar do próprio bolso. 

Por isso, **o Recall alto (92%) é a coroa de vitória deste projeto!**

---
*Salve este documento e o releia antes de fazer novos projetos de Machine Learning!*
