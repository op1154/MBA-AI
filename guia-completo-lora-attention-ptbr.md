# 🧠 Guia Completo em Markdown: LoRA + Attention em LLMs

> Material único em **Markdown**, em **português do Brasil**, reunindo **explicação conceitual**, **matemática**, **exemplos práticos de código** e uma **estrutura pronta para publicação no GitHub**.

---

## 📚 Sumário

1. [Visão geral](#-visão-geral)
2. [Parte 1 — Entendendo LoRA](#-parte-1--entendendo-lora)
3. [Parte 2 — Matemática do Attention](#-parte-2--matemática-do-attention)
4. [Exemplos práticos de código com LoRA](#-exemplos-práticos-de-código-com-lora)
5. [Exemplo prático da matemática do Attention em Python](#-exemplo-prático-da-matemática-do-attention-em-python)
6. [Estrutura sugerida para GitHub](#-estrutura-sugerida-para-github)
7. [Narrativa para portfólio](#-narrativa-para-portfólio)
8. [Conclusão](#-conclusão)

---

## ✨ Visão geral

Este material combina dois conceitos fundamentais dos **Large Language Models (LLMs)**:

- **LoRA (Low-Rank Adaptation)** → técnica para fazer fine-tuning eficiente, com menos custo computacional e menos memória.
- **Attention (Query, Key e Value)** → mecanismo que permite ao modelo entender o contexto e a relevância entre palavras.

A ideia aqui é unir:

- explicação didática,
- profundidade técnica,
- exemplos reproduzíveis em Python,
- e estrutura adequada para publicação no GitHub.

---

# 🚀 Parte 1 — Entendendo LoRA

## O que é LoRA?

**LoRA (Low-Rank Adaptation)** é uma técnica de adaptação eficiente de modelos grandes.

Em vez de atualizar **todos os pesos do modelo** durante o fine-tuning, o LoRA:

- mantém os pesos originais **congelados**;
- adiciona pequenas matrizes treináveis ao lado das projeções originais;
- aprende apenas a “correção” necessária para a nova tarefa.

### Em termos simples

```text
LoRA = pesos originais congelados + pequena atualização low-rank
```

Isso torna o fine-tuning:

- mais barato;
- mais rápido;
- mais leve para armazenar;
- mais fácil de compartilhar.

---

## Por que o fine-tuning completo é caro?

Modelos modernos possuem **bilhões de parâmetros**.

Quando fazemos **full fine-tuning**, precisamos atualizar todos esses parâmetros. Isso gera alguns problemas práticos:

1. **Uso elevado de memória de GPU** — precisamos armazenar pesos, gradientes e estados do otimizador.
2. **Treinamento mais lento** — atualizar bilhões de parâmetros exige muito processamento.
3. **Armazenamento pesado** — cada versão ajustada do modelo vira um checkpoint enorme.
4. **Distribuição difícil** — compartilhar um modelo ajustado completo é muito mais caro do que compartilhar apenas um adapter.

Por isso, o LoRA se tornou uma solução muito relevante no ecossistema de LLMs.

---

## A ideia central do LoRA

Imagine uma matriz de pesos original `W`.

No fine-tuning completo, faríamos algo assim:

```text
W_novo = W + ΔW
```

No LoRA, em vez de aprender uma matriz inteira `ΔW`, aprendemos a decomposição:

```text
ΔW = B × A
```

Onde:

- `A` é uma matriz pequena;
- `B` é outra matriz pequena;
- `r` é o rank (um número pequeno, como 4, 8, 16, 32 ou 64).

### Fórmula prática

```text
h = Wx + (α / r) × (BA)x
```

Onde:

- `W` = pesos originais (congelados)
- `A` e `B` = matrizes treináveis do LoRA
- `r` = rank
- `α` = fator de escala
- `x` = entrada da camada

---

## Intuição de negócio / produto

Uma boa analogia é pensar no modelo pré-treinado como um **livro gigantesco**.

- **Full fine-tuning** = reescrever várias páginas do livro.
- **LoRA** = manter o livro intacto e adicionar apenas pequenos post-its com os ajustes necessários.

Ou seja: o conhecimento base continua lá, e a adaptação fica concentrada em pequenos componentes reaproveitáveis.

---

## Exemplo numérico simples de economia

Suponha uma matriz de pesos de dimensão:

```text
4096 × 4096 = 16.777.216 parâmetros
```

Agora aplicando LoRA com `r = 8`:

- `A` → `8 × 4096 = 32.768 parâmetros`
- `B` → `4096 × 8 = 32.768 parâmetros`
- Total → `65.536 parâmetros`

### Comparação

```text
Full fine-tuning: 16.777.216 parâmetros
LoRA (r=8):          65.536 parâmetros
```

Isso representa uma redução enorme na quantidade de parâmetros treináveis.

---

## Onde o LoRA é aplicado em Transformers?

Na prática, o LoRA costuma ser aplicado principalmente nas projeções da atenção:

- `W_Q` → Query
- `W_K` → Key
- `W_V` → Value
- `W_O` → Output

O uso mais comum em muitos setups é aplicar LoRA em:

- `q_proj`
- `v_proj`

Em alguns modelos, também se aplica em outras camadas lineares, dependendo do objetivo.

---

## Merge do LoRA no modelo

Depois do treinamento, podemos incorporar o adapter ao modelo original:

```text
W_merge = W + (α / r)(BA)
```

### Vantagens do merge

- uso em inferência sem precisar carregar separadamente o adapter;
- simplificação da implantação;
- nenhum overhead extra em tempo de inferência após a fusão.

---

## Casos de uso reais do LoRA

O LoRA é amplamente usado em cenários como:

- fine-tuning de LLMs open source em GPU única;
- adaptação para atendimento ao cliente;
- ajuste para geração de código;
- adaptação de estilo de escrita;
- domínios especializados (jurídico, médico, financeiro);
- personalização de modelos com baixo custo;
- base para **QLoRA** (LoRA + quantização).

---

# 🧮 Parte 2 — Matemática do Attention

## O que é Attention?

O mecanismo de **Attention** permite que cada palavra de uma sequência avalie quais outras palavras são mais relevantes para o seu contexto.

A fórmula principal do **Scaled Dot-Product Attention** é:

```text
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
```

### Onde:

- `Q` = Query
- `K` = Key
- `V` = Value
- `Kᵀ` = transposta de K
- `d_k` = dimensão dos vetores Key

---

## Intuição rápida

- **Query (Q)**: o que a palavra está “procurando”
- **Key (K)**: o que cada palavra “oferece”
- **Value (V)**: a informação que será combinada

O objetivo é descobrir **quanto cada palavra deve prestar atenção nas outras**.

---

## Exemplo base: “I love AI”

Vamos usar embeddings pequenos só para fins didáticos:

```text
"I"    → [1.0, 0.0, 1.0, 0.0]
"love" → [0.0, 1.0, 0.0, 1.0]
"AI"   → [1.0, 1.0, 0.0, 0.0]
```

### Matriz de entrada X

```text
X =
| 1.0  0.0  1.0  0.0 |
| 0.0  1.0  0.0  1.0 |
| 1.0  1.0  0.0  0.0 |
```

---

## Criando Q, K e V

Usamos três matrizes de pesos:

```text
Q = X × W_Q
K = X × W_K
V = X × W_V
```

### Matrizes de peso do exemplo

```text
W_Q =
| 1 0 1 |
| 0 1 0 |
| 1 0 0 |
| 0 1 1 |

W_K =
| 0 1 0 |
| 1 0 1 |
| 0 0 1 |
| 1 1 0 |

W_V =
| 1 0 0 |
| 0 1 0 |
| 0 0 1 |
| 1 1 0 |
```

### Resultado

```text
Q =
| 2 0 1 |
| 0 2 1 |
| 1 1 1 |

K =
| 0 1 1 |
| 2 1 1 |
| 1 1 1 |

V =
| 1 0 1 |
| 1 2 0 |
| 1 1 0 |
```

---

## Calculando os scores: `Q × Kᵀ`

```text
Scores =
| 1 5 3 |
| 3 3 3 |
| 2 4 3 |
```

Interpretação simples:

- “I” presta mais atenção em “love”
- “love” distribui atenção igualmente
- “AI” também dá maior peso a “love”

---

## Scaling

Como `d_k = 3`, temos:

```text
√3 ≈ 1.732
```

Então escalamos os scores:

```text
Scaled Scores =
| 0.577  2.887  1.732 |
| 1.732  1.732  1.732 |
| 1.155  2.309  1.732 |
```

### Por que fazemos isso?

Porque scores muito altos podem deixar o softmax excessivamente “agressivo”, gerando valores muito extremos. O scaling ajuda a manter a estabilidade do treinamento.

---

## Softmax

Aplicando softmax em cada linha:

```text
Attention Weights =
| 0.070  0.707  0.223 |
| 0.333  0.333  0.333 |
| 0.168  0.533  0.299 |
```

### Interpretação

- “I” direciona 70,7% da atenção para “love”
- “love” mantém distribuição uniforme
- “AI” direciona 53,3% da atenção para “love”

---

## Saída final

Agora multiplicamos os pesos de atenção por `V`:

```text
Output = Attention Weights × V
```

Resultado:

```text
Output =
| 1.000  1.637  0.070 |
| 1.000  1.000  0.333 |
| 1.000  1.365  0.168 |
```

Cada palavra passa a ter uma nova representação enriquecida pelo contexto das demais.

---

# 💻 Exemplos práticos de código com LoRA

## Instalação das bibliotecas

```bash
pip install transformers peft datasets accelerate bitsandbytes torch
```

---

## Exemplo 1 — Aplicando LoRA a um modelo causal

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### O que esse código faz?

- carrega um modelo base (`gpt2`)
- define a configuração do LoRA
- aplica adapters sobre camadas-alvo
- mostra quantos parâmetros continuam treináveis

---

## Exemplo 2 — Criando um dataset simples e treinando

```python
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

textos = [
    "LoRA reduz o custo de fine-tuning em modelos grandes.",
    "Attention calcula relevância entre tokens.",
    "Transformers usam Query, Key e Value.",
    "Adapters LoRA são leves e reutilizáveis.",
]

dataset = Dataset.from_dict({"text": textos})

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="./outputs/lora-model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("./outputs/lora-adapter")
tokenizer.save_pretrained("./outputs/lora-adapter")
```

---

## Exemplo 3 — Inferência com o adapter treinado

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "gpt2"
adapter_path = "./outputs/lora-adapter"

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

prompt = "Explique LoRA de forma simples em português do Brasil."
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=80)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Exemplo 4 — Fazendo merge do adapter

```python
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import PeftModel

model_name = "gpt2"
adapter_path = "./outputs/lora-adapter"
output_dir = Path("./outputs/merged-model")
output_dir.mkdir(parents=True, exist_ok=True)

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, adapter_path)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_dir)
```

---

## Exemplo 5 — QLoRA (LoRA + quantização)

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "gpt2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

Depois disso, o LoRA pode ser aplicado normalmente com `get_peft_model(...)`.

---

# 🧪 Exemplo prático da matemática do Attention em Python

O código abaixo reproduz o exemplo didático de `Q`, `K`, `V`, scores, scaling, softmax e saída final.

```python
import math
import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


X = np.array([
    [1.0, 0.0, 1.0, 0.0],  # I
    [0.0, 1.0, 0.0, 1.0],  # love
    [1.0, 1.0, 0.0, 0.0],  # AI
])

W_Q = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
])

W_K = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 0, 1],
    [1, 1, 0],
])

W_V = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
])

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T
scaled_scores = scores / math.sqrt(K.shape[1])
attention_weights = softmax(scaled_scores)
output = attention_weights @ V

np.set_printoptions(precision=3, suppress=True)
print("Q:\n", Q)
print("\nK:\n", K)
print("\nV:\n", V)
print("\nScores:\n", scores)
print("\nScaled Scores:\n", scaled_scores)
print("\nAttention Weights:\n", attention_weights)
print("\nOutput:\n", output)
```

---

# 🧱 Estrutura sugerida para GitHub

Se você quiser transformar esse conteúdo em um repositório, esta estrutura funciona muito bem:

```text
meu-projeto-lora-attention/
├── README.md
├── requirements.txt
├── src/
│   ├── train_lora.py
│   ├── inference.py
│   ├── merge_adapter.py
│   └── attention_math_demo.py
├── docs/
│   └── portfolio-story.md
├── assets/
│   └── cover.svg
└── notebooks/
    └── exploracao.ipynb
```

---

# 🌟 Narrativa para portfólio

Você pode apresentar esse material como um projeto que demonstra:

- domínio conceitual de **Transformers**, **Attention**, **LoRA** e **PEFT**;
- habilidade de traduzir conceitos complexos em linguagem acessível;
- capacidade de conectar teoria, matemática e implementação prática;
- cuidado com documentação e apresentação técnica.

### Exemplo de narrativa curta para LinkedIn ou entrevista

> Estruturei um material técnico em português do Brasil para explicar, com profundidade conceitual e exemplos reproduzíveis em Python, dois fundamentos centrais dos LLMs modernos: o mecanismo de Attention e a técnica LoRA para fine-tuning eficiente. O objetivo foi unir clareza didática, matemática aplicada e organização de portfólio pronta para GitHub.

---

# ✅ Conclusão

Ao reunir **LoRA + Attention** em um único documento, você cria um material muito forte para:

- estudo técnico;
- publicação no GitHub;
- uso em aulas, workshops ou apresentações;
- fortalecimento de portfólio profissional.

### Em resumo

- **LoRA** resolve o problema de adaptação eficiente
- **Attention** resolve o problema de contexto
- **Juntos**, eles representam duas peças centrais nos LLMs modernos

---

## 📌 Sugestão de nome de arquivo

```text
guia-completo-lora-attention-ptbr.md
```

---

## 📄 Licença sugerida

Se publicar no GitHub, você pode usar a licença MIT para deixar o material reaproveitável.
