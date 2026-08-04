from src import config
from pymilvus import MilvusClient

client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)

if client.has_collection(config.MILVUS_COLLECTION_NAME):
    client.drop_collection(config.MILVUS_COLLECTION_NAME)
    print(f"Dropped collection: {config.MILVUS_COLLECTION_NAME}")
else:
    print(f"Collection {config.MILVUS_COLLECTION_NAME} does not exist -- nothing to drop.")