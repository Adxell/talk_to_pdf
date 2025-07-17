from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector

from langchain.embeddings.google_palm import GooglePalmEmbeddings
from config import settings

loader = DirectoryLoader(
    path="../pdfs",
    glob="**/*.pdf",
    use_multithreading=True, 
    show_progress=True,
    max_concurrency=50, 
    loader_cls=UnstructuredPDFLoader
)

docs = loader.load()

embeddings = GooglePalmEmbeddings(google_api_key=settings.api_key_google_palm)