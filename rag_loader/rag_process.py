import os 

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores.pgvector import PGVector

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
from langchain_experimental.text_splitter import SemanticChunker

loader = DirectoryLoader(
    path=os.path.abspath("../backend/pdfs/"),
    glob="**/*.pdf",
    use_multithreading=True, 
    show_progress=True,
    max_concurrency=50,
    loader_cls= PyPDFLoader
)

docs = loader.load()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07", 
                                          google_api_key=settings.GOOGLE_API_KEY)

text_splitter = SemanticChunker(
    embeddings=embeddings,
)
flattend_docs = text_splitter.split_documents(docs)


PGVector.from_documents(
    documents=flattend_docs,
    embedding=embeddings,
    collection_name="pdfs_collection",
    connection_string=settings.PGVECTOR_CONNECTION_STRING,
    pre_delete_collection=True
)   