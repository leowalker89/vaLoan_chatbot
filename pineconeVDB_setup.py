from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.pinecone import Pinecone
import os
from dotenv import load_dotenv
import pinecone

load_dotenv()

documents = []

for file in os.listdir("pdfs"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader("pdfs/" + file)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

index_name = "va-loans"

Pinecone.from_documents(docs, embeddings, index_name=index_name)



