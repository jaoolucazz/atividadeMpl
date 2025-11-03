# Perceptron Multicamadas do Zero - Porta Lógica XOR

Este repositório contém uma implementação de uma rede neural do tipo **Perceptron Multicamadas (MLP)** para resolver o problema da **porta lógica XOR**. A implementação foi feita do zero em Python, sem o uso de bibliotecas específicas de redes neurais, exceto **NumPy** para cálculos matemáticos e operações de matriz.

## Objetivo

O objetivo deste código é demonstrar como construir e treinar uma rede neural simples com uma camada oculta para aprender a função XOR. A função XOR é um problema clássico de aprendizado de máquina, onde a saída é `1` quando as entradas são diferentes e `0` quando são iguais. A rede é treinada usando o algoritmo **Backpropagation**.

## Como funciona

### Arquitetura da Rede Neural
A rede neural tem a seguinte arquitetura:

- **Entrada**: 2 neurônios (correspondentes às duas entradas da porta XOR).
- **Camada Oculta**: 2 neurônios.
- **Saída**: 1 neurônio (a saída da porta XOR).

### Função de Ativação
A rede usa a função de ativação **sigmoide** (ou logística) para as unidades da camada oculta e a unidade de saída.

### Algoritmo de Treinamento
O treinamento é feito por **backpropagation**, ajustando os pesos com base no erro entre a saída desejada e a saída prevista. A função de erro usada é o erro quadrático médio, e os pesos são ajustados com base na derivada da função de ativação.
