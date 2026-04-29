import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

#                   -------1. Carga de configuración-------
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
base_url = os.getenv("OPENAI_BASE_URL")
embeddings_url = os.getenv("OPENAI_EMBEDDINGS_URL")

st.set_page_config(page_title="Asistente Pesquero Biobío", layout="wide")
st.title("Gestión Pesquera del Biobío")
st.markdown("---")

#                   -------2. Motor RAG-------
@st.cache_resource
def iniciar_sistema_pesquero():
    data_folder = "data/"
    docs = []
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    for archivo in os.listdir(data_folder):
        if archivo.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, archivo))
            docs.extend(loader.load())
    
    if not docs:
        st.error("No se encontraron PDFs en la carpeta /data")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    trozos = splitter.split_documents(docs)
    
    #                   ------- Configuración de Embeddings para GitHub Models -------
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        api_key=token, 
        base_url=embeddings_url
    )
    vector_db = FAISS.from_documents(trozos, embeddings)
    return vector_db

#                   -------3. Diseño del Prompt (IE2) ------
template = """Eres un experto en leyes pesqueras chilenas (Ley 18.892) y seguridad naval. 
Usa el siguiente contexto para responder la duda del trabajador de forma profesional.

CONTEXTO: {context}
PREGUNTA: {question}

Si la información no está en los documentos, responde: "No dispongo de esa información oficial para evitar riesgos legales". No inventes datos."""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

#                   -------4. Ejecución del sistema-------
if token:
    db = iniciar_sistema_pesquero()
    if db:
        #                   ------- Configuración de LLM para GitHub Models -------
        llm = ChatOpenAI(
            model="gpt-4o", 
            api_key=token, 
            base_url=base_url, 
            temperature=0
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

        query = st.chat_input("Consulta la normativa pesquera...")
        if query:
            with st.chat_message("user"): st.write(query)
            with st.chat_message("assistant"):
                respuesta = chain.invoke(query)["result"]
                st.write(respuesta)
else:
    st.error("Falta el GITHUB_TOKEN en el archivo .env")