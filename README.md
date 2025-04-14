
---

# 🧪 Sistema RAG com Avaliação Automatizada para Físico-Química

Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation)** aplicado a documentos acadêmicos de **físico-química**, utilizando **FAISS**, **Sentence Transformers** e **modelos LLM via OpenRouter**, com avaliação automatizada de qualidade das respostas geradas.

---

## ⚙️ Tecnologias Utilizadas

- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenRouter](https://openrouter.ai/) (LLM backend)
- `dotenv`, `pickle`, `matplotlib`, `pandas`
- Python 3.10+

---

## 📐 Arquitetura do Sistema

### 🔹 Chunking
Documentos extensos são divididos em **chunks** (blocos de texto menores), geralmente com ~300 palavras, para facilitar a indexação e o processamento pelos modelos.

### 🔹 Embeddings e Busca Semântica
1. Cada chunk e cada pergunta são convertidos em vetores com **Sentence Transformers**.
2. Esses vetores são indexados no **FAISS** para busca por similaridade vetorial.
3. Os **TOP_K (ex: 30)** chunks mais similares à pergunta são recuperados.

### 🔹 Prompt e Geração de Resposta
- Um prompt especializado orienta o modelo LLM a responder apenas com base no **contexto fornecido**, evitando alucinações e mantendo linguagem científica precisa.
- O modelo utilizado pode ser, por exemplo, `"mistralai/mistral-7b-instruct"` via OpenRouter.

### 🔹 Avaliação Automatizada
As respostas geradas são avaliadas automaticamente com base em:
- Presença de termos específicos (ex: “valor”, “igual a”)
- Comprimento da resposta
- Expressões de incerteza

Classificação: ✅ **Correta**, ⚠️ **Parcial**, ❌ **Incorreta**


### 🔹 Exemplo de prompt auxiliar:

    """Você é um especialista altamente técnico em físico-química,  responsável por responder com base **apenas** no conteúdo do     material fornecido.

    🔹 Utilize **somente informações literalmente contidas no contexto**.
    🔹 Responda com precisão, terminologia científica e sem interpretações externas.
    🔹 Se a resposta não puder ser obtida diretamente do conteúdo, diga claramente:
    “Informação não encontrada no material.”

    ━━━━━━━━ CONTEXTO ━━━━━━━━
      {context}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━

     ❓ Pergunta:
    {question}

    ✍️ Regras:
    - Não utilize conhecimento externo.
    - Não repita a pergunta.
    - Seja técnico, objetivo e direto.
    - Sempre fundamente a resposta com evidência textual, se possível.

---

## 📂 Estrutura de Arquivos

| Arquivo                 | Descrição |
|-------------------------|-----------|
| `index.faiss`           | Índice vetorial gerado com embeddings dos chunks |
| `index.pkl`             | Lista de chunks originais e seus metadados |
| `.env`                  | Contém a chave `OPENROUTER_API_KEY` |
| `avaliacao_rag_resultados.csv` | Resultados salvos com pergunta, resposta, avaliação e contexto |
| `grafico_rag_resultado.png`    | Gráfico com distribuição das respostas |

---

## 📊 Visualização de Resultados

O script gera um gráfico de barras com as quantidades de respostas classificadas como **corretas**, **parciais** e **incorretas**, além de exportar um relatório em CSV com os dados completos de avaliação.

---

## 🚀 Como Executar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Configure a chave de API no arquivo `.env`:

```env
OPENROUTER_API_KEY=sua_chave_aqui
```

3. Execute o script de avaliação:

```bash
python avaliar_rag.py
```

---

## 📌 Variáveis Importantes

| Variável       | Função |
|----------------|--------|
| `TOP_K`        | Número de chunks mais relevantes usados no contexto |
| `MODEL_NAME`   | Modelo de embeddings (ex: `intfloat/e5-base-v2`) |
| `MODEL`        | Nome do modelo LLM via OpenRouter |
| `PROMPT_BASE`  | Prompt que guia o comportamento da IA |

---

## ✅ Pontos Fortes do Projeto

- ⚡ **Busca semântica eficiente** com FAISS
- 🎯 **Geração controlada** via prompt técnico
- 🧪 **Avaliação automatizada** de performance
- 🔁 Fácil adaptação para diferentes domínios e LLMs

---

## 📬 Contato

Desenvolvido por [Seu Nome Aqui]  
Email: [seu@email.com]  
LinkedIn: [linkedin.com/in/seu-perfil]

---