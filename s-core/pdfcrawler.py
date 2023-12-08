from elasticsearch import Elasticsearch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings

from pathlib import Path

ES_URL = 'https://127.0.0.1:9200'
ES_USER = "elastic"
ES_USER_PASSWORD = "elastic"
CERT_PATH = 'D:\\es\\8.11.1\\kibana-8.11.1\\data\\ca_1701918227592.crt'


pdfs = []
docs = []

text_splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=256
)

for f in Path("data").iterdir():
    if f.is_file():
        pdfs.append(f)
        # print(str(f))
        loader = PyPDFLoader(file_path=str(f))
        pages = loader.load_and_split(text_splitter)
        # print(pages)
        docs += pages

print(f"Split {len(pdfs)} documents into {len(docs)} passages")

embeddings = HuggingFaceEmbeddings(model_name=str(Path("models/multilingual-e5-base")), model_kwargs = {'device': 'cpu'} )

client = Elasticsearch(
    ES_URL,
    basic_auth=(ES_USER, ES_USER_PASSWORD),
    ca_certs=CERT_PATH
)

print("Connection Success!")

vector_store = ElasticsearchStore.from_documents(
    docs,
    es_connection = client,
    index_name="es-docs",
    embedding=embeddings
)

print("Store Success!")
