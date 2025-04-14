# Substitua o conteúdo do seu arquivo atual com o código abaixo

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredEPubLoader,
    PyMuPDFLoader,
)
import os
import pickle
import shutil
from dotenv import load_dotenv
import pandas as pd
import pypandoc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === Carregar variáveis de ambiente ===
load_dotenv()

# === Configurações ===
PASTA_DOCUMENTOS = "docs"
OUTPUT_FAISS = "meu_indice"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 75
EMBEDDING_MODEL = "intfloat/e5-base-v2"
MIN_CHUNK_LENGTH = 25
ERROS = []

# === Verificação de dependências ===
def verificar_dependencias():
    print("\n🔎 Verificando dependências...")
    if shutil.which("pdftotext"):
        print("✅ Poppler (pdftotext) encontrado no PATH.")
    else:
        print("❌ Poppler não encontrado. Verifique se 'pdftotext' está no PATH.")

    try:
        pandoc_path = pypandoc.get_pandoc_path()
        print(f"✅ Pandoc encontrado em: {pandoc_path}")
    except Exception as e:
        print("⚠️ Pandoc não encontrado. Tentando baixar...")
        try:
            pypandoc.download_pandoc()
            print("✅ Pandoc baixado com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao baixar pandoc: {e}")

# === Função para carregar documentos ===
def carregar_documentos(pasta):
    documentos = []
    for nome_arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, nome_arquivo)
        try:
            if nome_arquivo.endswith((".txt", ".md")):
                loader = TextLoader(caminho)
            elif nome_arquivo.endswith(".pdf"):
                loader = PyMuPDFLoader(caminho)
            elif nome_arquivo.endswith(".epub"):
                try:
                    loader = UnstructuredEPubLoader(caminho)
                except Exception as epub_err:
                    try:
                        print(f"⚠️ UnstructuredEPubLoader falhou, tentando via pandoc: {nome_arquivo}")
                        text = pypandoc.convert_file(caminho, "plain")
                        with open(caminho + ".txt", "w", encoding="utf-8") as f:
                            f.write(text)
                        loader = TextLoader(caminho + ".txt")
                    except Exception as e:
                        raise RuntimeError(f"Erro ao tentar fallback EPUB com pandoc: {e}")
            else:
                loader = UnstructuredFileLoader(caminho)

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = nome_arquivo
            documentos.extend(docs)
            print(f"✅ Carregado: {nome_arquivo} ({len(docs)} docs)")
        except Exception as e:
            print(f"❌ Erro ao carregar {nome_arquivo}: {e}")
            ERROS.append((nome_arquivo, str(e)))
    return documentos

# === Split com remoção de duplicatas e ajuste de tamanho ===
def dividir_documentos(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip().split()) >= MIN_CHUNK_LENGTH]
    chunk_dict = {}
    for c in chunks:
        texto = c.page_content.strip()
        if texto not in chunk_dict:
            chunk_dict[texto] = c
    return list(chunk_dict.values())

# === Criar e salvar índice FAISS ===
def criar_indice_faiss(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    index.save_local(OUTPUT_FAISS)
    with open(os.path.join(OUTPUT_FAISS, "index.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    pd.DataFrame([{"chunk": doc.page_content, **doc.metadata} for doc in chunks]).to_csv("chunks_exportados.csv", index=False)
    print(f"\n✅ Índice salvo em: {OUTPUT_FAISS}/index.faiss e index.pkl")
    print("📝 Exportado: chunks_exportados.csv")

# === Pipeline ===
def gerar_indice():
    verificar_dependencias()
    print("📥 Carregando documentos...")
    docs = carregar_documentos(PASTA_DOCUMENTOS)
    if not docs:
        print("🚫 Nenhum documento carregado.")
        return
    print("\n✂️ Dividindo documentos em chunks...")
    chunks = dividir_documentos(docs)
    print(f"🔹 Total de chunks válidos (sem duplicatas): {len(chunks)}")
    print("\n🧠 Gerando embeddings e salvando índice...")
    criar_indice_faiss(chunks)

    if ERROS:
        print("\n⚠️ Alguns arquivos falharam ao serem carregados:")
        for nome, erro in ERROS:
            print(f"  - {nome}: {erro}")

# === Execução ===
if __name__ == "__main__":
    gerar_indice()
