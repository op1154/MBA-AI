# 🚀 Guia Completo: Query (Q), Key (K) e Value (V) em Transformers

## 📖 Introdução

O mecanismo de **Self-Attention** é o coração dos modelos Transformer.
Ele permite que cada palavra analise o contexto das demais palavras
antes de produzir sua representação final.

------------------------------------------------------------------------

# 📚 Exemplo: "Eu amo IA"

## 1. Embeddings

  Palavra   Embedding
  --------- ------------------------
  **Eu**    `[1.0, 0.0, 1.0, 0.0]`
  **amo**   `[0.0, 1.0, 0.0, 1.0]`
  **IA**    `[1.0, 1.0, 0.0, 0.0]`

Matriz `X`:

``` text
| 1.0  0.0  1.0  0.0 |
| 0.0  1.0  0.0  1.0 |
| 1.0  1.0  0.0  0.0 |
```

------------------------------------------------------------------------

## 2. Gerando Q, K e V

Fórmulas:

``` text
Q = X × WQ
K = X × WK
V = X × WV
```

### Resultado

  Palavra   Query (Q)   Key (K)     Value (V)
  --------- ----------- ----------- -----------
  Eu        `[2,0,1]`   `[0,1,1]`   `[1,0,1]`
  amo       `[0,2,1]`   `[2,1,1]`   `[1,2,0]`
  IA        `[1,1,1]`   `[1,1,1]`   `[1,1,0]`

### Intuição

  Matriz          Função
  --------------- --------------------------------------
  **Query (Q)**   O que esta palavra procura?
  **Key (K)**     Que informação esta palavra oferece?
  **Value (V)**   Qual conteúdo será compartilhado?

------------------------------------------------------------------------

## 3. Cálculo dos Escores

``` text
Scores = Q × Kᵀ
```

  Origem \\ Destino     Eu   amo   IA
  ------------------- ---- ----- ----
  **Eu**                 1     5    3
  **amo**                3     3    3
  **IA**                 2     4    3

Interpretação: **"Eu"** possui maior afinidade com **"amo"**.

------------------------------------------------------------------------

## 4. Escalonamento

Divisão por:

``` text
√dₖ = √3 ≈ 1,732
```

  Origem         Eu     amo      IA
  --------- ------- ------- -------
  **Eu**      0,577   2,887   1,732
  **amo**     1,732   1,732   1,732
  **IA**      1,155   2,309   1,732

------------------------------------------------------------------------

## 5. Softmax

Após aplicar Softmax:

  Palavra     Atenção para Eu   Atenção para amo   Atenção para IA
  --------- ----------------- ------------------ -----------------
  **Eu**                 7,0%          **70,7%**             22,3%
  **amo**               33,3%              33,3%             33,3%
  **IA**                16,8%          **53,3%**             29,9%

Cada linha soma 100%.

------------------------------------------------------------------------

## 6. Saída Final

``` text
Output = Softmax(QKᵀ / √dₖ) × V
```

  Palavra   Vetor Resultante
  --------- -------------------------
  **Eu**    `[1.000, 1.637, 0.070]`
  **amo**   `[1.000, 1.000, 0.333]`
  **IA**    `[1.000, 1.365, 0.168]`

------------------------------------------------------------------------

# 🔄 Fluxo Completo

``` text
Texto
   │
   ▼
Embeddings (X)
   │
   ▼
Q = XWQ
K = XWK
V = XWV
   │
   ▼
Q × Kᵀ
   │
   ▼
Dividir por √dₖ
   │
   ▼
Softmax
   │
   ▼
Pesos de Atenção
   │
   ▼
Multiplicar por V
   │
   ▼
Representação Final Contextualizada
```

------------------------------------------------------------------------

# 📌 Resumo

  Etapa   Operação
  ------- ----------------------------------
  1       Converter palavras em embeddings
  2       Calcular Q, K e V
  3       Computar `Q × Kᵀ`
  4       Escalar por `√dₖ`
  5       Aplicar Softmax
  6       Multiplicar por `V`
  7       Obter vetores contextualizados

## Fórmula principal

``` text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

Essa operação é executada bilhões de vezes em Grandes Modelos de
Linguagem modernos, permitindo compreender contexto, relações semânticas
e dependências entre palavras.
