# Import modules
from pathlib import Path
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama

DIRECTORY="/mnt/p/Mes Documents/Assurance/accident scooter octobre 2023/expertise/"
reader = SimpleDirectoryReader(input_dir=DIRECTORY)
documents = reader.load_data()

# Create Qdrant client and store
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="pds")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize Ollama and ServiceContext
llm = Ollama(model="mistral:latest")
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

# Create VectorStoreIndex and query engine
index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
query_engine = index.as_query_engine()

# Perform a query and print the response
response = query_engine.query("Qui suit le dossier relatif à l'accident de scooter chez l'exoert?")
print(response)
