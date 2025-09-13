# db_astra.py
from typing import Iterable, List
from config import settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Cassandra
import cassio

def get_session():
    cassio.init(
        token=settings.ASTRA_DB_APPLICATION_TOKEN,
        database_id=settings.ASTRA_DB_ID,
        keyspace=settings.ASTRA_DB_KEYSPACE,
    )
    return cassio

def get_embeddings():
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)

def get_vector_store() -> Cassandra:
    get_session()
    vs = Cassandra(
        embedding=get_embeddings(),
        table_name=settings.VECTOR_TABLE
    )
    return vs

def add_documents(docs: Iterable[str], metadatas: Iterable[dict] = None) -> List[str]:
    vs = get_vector_store()
    ids = vs.add_texts(texts=list(docs), metadatas=list(metadatas) if metadatas else None)
    return ids
