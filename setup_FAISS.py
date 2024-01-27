from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def create_or_get_vector_store() -> FAISS:
    """
    Creates or loads the vector database locally

    Args:
        chunks: List of chunks

    Returns:
        FAISS: Vector store
    """

    embeddings = OpenAIEmbeddings() # we can change it at will!
    # embeddings = HuggingFaceInstructEmbeddings() # for example by uncommenting here and commenting the line above

    if not os.path.exists("./db"):
        documents = []

        for file in os.listdir("pdfs"):
            if file.endswith(".pdf"):
                loader = PyPDFLoader("pdfs/" + file)
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        print("CREATING DB")
        vectorstore = FAISS.from_documents(
            docs, embeddings
        )
        vectorstore.save_local("./db")
    else:
        print("LOADING DB")
        vectorstore = FAISS.load_local("./db", embeddings)

    return vectorstore