# 🧠 Ejemplo RAG con Amazon Bedrock + Streamlit

Este proyecto demuestra cómo construir un sistema **RAG (Retrieval-Augmented Generation)** simple usando:

- **Amazon Bedrock**, con:
  - `amazon.titan-embed-text-v1` para embeddings
  - `anthropic.claude-v2` y `amazon.titan-text-lite-v1` para generación de texto
- **Streamlit** como interfaz interactiva
- **FAISS** como vector store local

---

## 🚀 Cómo ejecutar

```bash
# 1. Clona el repositorio
git clone https://github.com/lobolc/rag-bedrock-ejemplo.git
cd rag-bedrock-ejemplo

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Configura tus credenciales de AWS
aws configure

# 4. Agrega tus archivos PDF
mkdir -p data
cp archivo.pdf data/

# 5. Ejecuta la aplicación
streamlit run app/main.py
