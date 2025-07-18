import os 
from operator import itemgetter

from config import settings

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


vector_store = PGVector(
    embedding_function=GoogleGenerativeAIEmbeddings(),
    collection_name="pdfs_collection",
    connection_string=settings.PGVECTOR_CONNECTION_STRING
)

template = """
answer the question based on the context below. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.0,
    max_output_tokens=1024,
    streaming=True
)

chain = (
    prompt | vector_store | llm
) | StrOutputParser()