import boto3
import streamlit as st

# Updated import for BedrockEmbeddings
from langchain_aws import BedrockEmbeddings

# Bedrock LLM (still under community)
from langchain_community.llms import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector store
from langchain_community.vectorstores import FAISS

# LangChain core
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Embeddings model (usually available)
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

# Step 1: PDF Ingestion

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs

# Step 2: Vector Store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# Step 3: LLM Loaders

def get_claude_llm():
    # Anthropic Claude v2
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=bedrock,
        model_kwargs={"max_tokens_to_sample": 512}
    )
    return llm

def get_titan_llm():
    # Amazon Titan Text Lite
    llm = Bedrock(
        model_id="amazon.titan-text-lite-v1",
        client=bedrock,
        model_kwargs={"maxTokenCount": 512}
    )
    return llm

# Step 4: Prompt Template

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end. Summarize with 
**at least 250 words** and include detailed explanations. 
If you don't know the answer, just say you don't know—do not make one up.

<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Step 5: QA Function

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer["result"]

# Step 6: Streamlit App

def main():
    st.set_page_config(page_title="Chat PDF")

    st.header("Chat with PDF using AWS Bedrock 💁")

    user_question = st.text_input("Ask a question about the uploaded PDF files:")

    with st.sidebar:
        st.title("Update or Create Vector Store")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated!")

    if st.button("Claude Output"):
        with st.spinner("Generating answer with Claude..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Answer generated!")

    if st.button("Titan Output"):
        with st.spinner("Generating answer with Titan..."):
            faiss_index = FAISS.load_local(
                "faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True
            )
            llm = get_titan_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Answer generated!")

# 🟢 Entry Point
if __name__ == "__main__":
    main()

