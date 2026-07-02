import os
import lancedb

from config import *
    
LANCEDB_BUCKET = "veradoc-datalake/vector_indices"
LANCEDB_TABLE = "document_embeddings"

# connect to lancedb + minio
lancedb_uri = f"s3://{LANCEDB_BUCKET}"

storage_options = {
    "endpoint": os.getenv("MINIO_ENDPOINT_URL"),
    "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),        
    "allow_http": "true",
    "region": "us-east-1"
}

print(storage_options)

db_connection = lancedb.connect(
    lancedb_uri,
    storage_options=storage_options
)

# Connect to your datalake table
table = db_connection.open_table(LANCEDB_TABLE)

# Select all 'id' column and convert to a list or dataframe
ids = table.search() \
    .select(["id"]) \
    .to_pandas()

print("Actual Record IDs in this table:")
print(ids)

# Querying by a specific ID or metadata property
result_df = table.search() \
    .where("id='538aedc6-23ff-4879-800d-fbb8e41e3bba'") \
    .to_pandas()

# Inspecting the register(lancedb row)
if not result_df.empty:
    chunk = result_df["text"].values[0]       # Assuming 'text' is your chunk column
    embedding = result_df["vector"].values[0] # 'vector' is the default embedding column
    
    print(f"Chunk text:\n{chunk}\n")
    print(f"Embedding vector (Length {len(embedding)}):\n{embedding}")
else:
    print("Register not found.")