from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector

from langchain.embeddings.google_palm import GooglePalmEmbeddings
from config import settings
from langchain_experimental.text_splitter import SemanticChunker

loader = DirectoryLoader(
    path="../pdfs",
    glob="**/*.pdf",
    use_multithreading=True, 
    show_progress=True,
    max_concurrency=50, 
    loader_cls=UnstructuredPDFLoader
)

docs = loader.load()

embeddings = GooglePalmEmbeddings(google_api_key=settings.API_KEY_GOOGLE_PALM)

text_splitter = SemanticChunker(
    embeddings=embeddings,
)
flattend_docs =  [doc[0] for doc in docs if doc]
flattend_docs = text_splitter.split_documents(flattend_docs)

PGVector.from_documents(
    documents=flattend_docs,
    embeddings=embeddings,
    collection_name="pdfs_collection",
    connection_string=settings.PGVECTOR_CONNECTION_STRING,
    pre_delete_collection=True
)   