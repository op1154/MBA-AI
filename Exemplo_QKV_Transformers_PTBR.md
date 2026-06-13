# Exemplo Prático: Query (Q), Key (K) e Value (V) em Transformers

## Passo 1: Convertendo palavras em embeddings

Frase:

    Eu amo IA

Embeddings simplificados:

  Palavra   Vetor
  --------- ------------------------
  Eu        `[1.0, 0.0, 1.0, 0.0]`
  amo       `[0.0, 1.0, 0.0, 1.0]`
  IA        `[1.0, 1.0, 0.0, 0.0]`

Matriz de entrada `X`:

              d_emb = 4
            ┌─────────────────────┐
    Eu   →  │ 1.0  0.0  1.0  0.0 │
    amo  →  │ 0.0  1.0  0.0  1.0 │
    IA   →  │ 1.0  1.0  0.0  0.0 │
            └─────────────────────┘

## Passo 2: Criando Q, K e V

    Q = X × WQ
    K = X × WK
    V = X × WV

Exemplo de matrizes:

    WQ =
    |1 0 1|
    |0 1 0|
    |1 0 0|
    |0 1 1|

    WK =
    |0 1 0|
    |1 0 1|
    |0 0 1|
    |1 1 0|

    WV =
    |1 0 0|
    |0 1 0|
    |0 0 1|
    |1 1 0|

Resultados:

  Palavra   Q           K           V
  --------- ----------- ----------- -----------
  Eu        `[2,0,1]`   `[0,1,1]`   `[1,0,1]`
  amo       `[0,2,1]`   `[2,1,1]`   `[1,2,0]`
  IA        `[1,1,1]`   `[1,1,1]`   `[1,1,0]`

## Passo 3: Escores de Atenção

    Scores = Q × Kᵀ

          Eu   amo   IA
  ----- ---- ----- ----
  Eu       1     5    3
  amo      3     3    3
  IA       2     4    3

## Passo 4: Escalonamento

Divide-se cada valor por `√dₖ`, onde `dₖ = 3`:

`√3 ≈ 1,732`

## Passo 5: Softmax

Pesos de atenção:

             Eu     amo      IA
  ----- ------- ------- -------
  Eu      0,070   0,707   0,223
  amo     0,333   0,333   0,333
  IA      0,168   0,533   0,299

## Passo 6: Saída Final

    Saída = Attention Weights × V

  Palavra   Novo vetor
  --------- -------------------------
  Eu        `[1.000, 1.637, 0.070]`
  amo       `[1.000, 1.000, 0.333]`
  IA        `[1.000, 1.365, 0.168]`

## Resumo

A fórmula central do mecanismo de atenção é:

    Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V

Esse processo permite que cada palavra incorpore contexto das demais
palavras da sequência, sendo a base dos modelos Transformer modernos.
