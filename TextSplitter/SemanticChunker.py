from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# RAG를 잘할 수 있는 비법!
# 텍스트를 의미론적으로 유사한 청크로 분할하는 역할
with open("./data/appendix-keywords.txt") as f:
    file = f.read()
    
load_dotenv()

text_splitter = SemanticChunker(OpenAIEmbeddings())
chunks = text_splitter.split_text(file)

print(chunks[0])

docs = text_splitter.create_documents([file])
print(docs[0].page_content)  