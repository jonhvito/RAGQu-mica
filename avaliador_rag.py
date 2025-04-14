import pickle
import faiss
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

# === ENV ===
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# === CONFIG ===
INDEX_PATH = "meu_indice/index.faiss"
PKL_PATH = "meu_indice/index.pkl"
MODEL_NAME = "intfloat/e5-base-v2"
TOP_K = 30
MODEL = "mistralai/mistral-7b-instruct"

# === CLIENT OPENROUTER ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# === PROMPT ===
PROMPT_BASE = """Você é um especialista altamente técnico em físico-química, responsável por responder com base **apenas** no conteúdo do material fornecido.

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
"""

# === PERGUNTAS BASEADAS NOS CHUNKS ===
PERGUNTAS = [
    "O que o material diz sobre equação de velocidade?",
    "O que o material diz sobre constante de equilíbrio?",
    "O que o material diz sobre primeira lei da termodinâmica?",
    "O que o material diz sobre energia de ativação?",
    "O que o material diz sobre célula eletroquímica?"
]

# === UTILITÁRIOS ===
def extrair_texto(entry):
    if isinstance(entry, dict):
        return entry.get("page_content") or entry.get("text") or ""
    elif hasattr(entry, 'page_content'):
        return entry.page_content
    elif isinstance(entry, str):
        return entry
    return ""

# === CARREGAR DADOS ===
print("🔍 Carregando índice e documentos...")
index = faiss.read_index(INDEX_PATH)
with open(PKL_PATH, "rb") as f:
    stored_data = pickle.load(f)
texts = [extrair_texto(entry) for entry in stored_data]
model = SentenceTransformer(MODEL_NAME)

# === AVALIAR RAG ===
resultados = []
print("\n🧪 Executando perguntas de teste...\n")
for idx, pergunta in enumerate(PERGUNTAS, 1):
    pergunta_vec = model.encode([pergunta])
    D, I = index.search(pergunta_vec, TOP_K)
    indices_validos = [i for i in I[0] if i < len(texts)]
    contexto = "\n---\n".join([texts[i] for i in indices_validos])
    prompt = PROMPT_BASE.format(context=contexto, question=pergunta)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Você é um assistente técnico especializado em físico-química."},
            {"role": "user", "content": prompt}
        ]
    )

    resposta = response.choices[0].message.content.strip()
    resposta_len = len(resposta.split())
    contexto_len = len(contexto.split())
    if "informação não encontrada" in resposta.lower():
        avaliacao = "Incorreta"
    elif resposta_len < 25:
        avaliacao = "Parcial"
    elif any(x in resposta.lower() for x in ["valor", "resposta correta", "é igual a"]):
        avaliacao = "Correta"
    else:
        avaliacao = "Parcial"

    resultados.append({
        "Pergunta": pergunta,
        "Resposta": resposta,
        "Avaliação": avaliacao,
        "Contexto": contexto,
        "Resposta_len": resposta_len,
        "Contexto_len": contexto_len,
        "Lacuna": "❗ Contexto curto" if contexto_len < 60 else ("⚠️ Resposta vaga" if resposta_len < 25 else "✅ OK")
    })

# === SALVAR CSV E MOSTRAR GRÁFICO ===
df = pd.DataFrame(resultados)
df.to_csv("avaliacao_rag_resultados.csv", index=False)

contagem = df["Avaliação"].value_counts()
plt.figure(figsize=(8, 5))
cores = {"Correta": "green", "Parcial": "orange", "Incorreta": "red"}
contagem.plot(kind="bar", color=[cores.get(x, "gray") for x in contagem.index])
plt.title("Distribuição de Respostas RAG")
plt.xlabel("Avaliação")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("grafico_rag_resultado.png")
plt.show()

# === MOSTRAR RESPOSTAS COM INFORMAÇÃO ===
print("\n📋 Resumo das respostas:\n")
with pd.option_context('display.max_colwidth', None):
    print(df[["Pergunta", "Avaliação", "Lacuna"]])
