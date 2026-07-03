# Description
PoC Veradoc, refactotr to use all llamaindex in the ingestion pipeline. Remove partial dependencies langchain communitiy. 
The purpose of this PoC is take a decision:

- Use llamaindex to implement the ingestion pipeline: less dependencies and better integration with embeddings models.
- Use langchain for agentic RAG

## Dependencies:

- llama-index: El núcleo del framework (gestiona los Nodes /Chunks), IngestionPipeline y TokenTextSplitter).
- llama-index-readers-file:  llamaindex reader file integration
- llama-index-readers-s3: Añade el S3Reader para conectar, descargar y mapear datos directamente desde tu MinIO local.
- llama-index-embeddings-ollama: El puente nativo para interactuar con la API local de Ollama de forma optimizada.
- llama-index-vector-stores-lancedb: El conector oficial para que el pipeline entienda cómo guardar las estructuras de datos en LanceDB.
- lancedb: El motor base de la base de datos vectorial serverless en formato Arrow.
- pypdf: La librería nativa que utiliza por debajo el extractor PDFReader() de LlamaIndex (sustituyendo a PyPDFLoader).
- pytesseract: to use the ocr tesseract installed locally
- aspose-words: tool to split office doc files



```shell
$ pip install python-dotenv \
              llama-index \
              llama-index-readers-file \
              llama-index-readers-s3 \
              llama-index-embeddings-ollama \
              llama-index-vector-stores-lancedb \
              lancedb \
              openpyxl \
              docx2txt \
              striprtf \
              aspose-words \
              aspose-slides \
              openpyxl \
              python-pptx \
              pypdf \                      
              xlrd \
              pytesseract \
              Pillow \
              EbookLib
```              