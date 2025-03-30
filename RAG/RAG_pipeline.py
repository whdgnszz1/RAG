from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import os

load_dotenv()
logging.langsmith("default")

# PDF 로드
loader = PyPDFLoader('https://wdr.ubion.co.kr/wowpass/img/event/gsat_170823/gsat_170823.pdf')
pages = loader.load_and_split()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splited_docs = text_splitter.split_documents(pages)

# 임베딩 모델
model_huggingface = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

# 기존 컬렉션 삭제 후 새로 생성
chroma_db = Chroma.from_documents(splited_docs, model_huggingface, persist_directory='./samsung_db')
chroma_db.persist()

# 문서 수 확인
print('문서의 수:', chroma_db._collection.count())

# 질의하기
question = '삼성전자의 주요 사업영역은?'
docs = chroma_db.similarity_search(question, k=4)

print('검색된 문서의 수:', len(docs))
for doc in docs:
    print(doc)
    print('--' * 100)
    
# GPT-4에 전달할 프롬프트
template = """당신은 삼성전자 기업 보고서를 설명해주는 챗봇 '삼성맨'입니다.
안상준 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 이모지를 사용하며 친근하게 답변하세요.
{context}

Question: {question}
Answer:
"""

# 프롬프트 템플릿은 {question}에는 사용자의 질문이 들어간다. {context}에는 사용자의 질문에 따른 검색 결과가 들어간다.
prompt = PromptTemplate.from_template(template)

# GPT-4 선언
llm = ChatOpenAI(model_name="gpt-4o")

# 벡터 데이터베이스를 LLM과 연결하기 위한 retriever 객체
retriever = chroma_db.as_retriever(search_kwargs={"k": 3})

# RetrievalQA 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # LLM 연결
    retriever=retriever,  # 리트리버(벡터 데이터베이스) 연결
    chain_type_kwargs={"prompt": prompt},  # 프롬프트 템플릿 연결
    return_source_documents=True  # 검색된 문서 반환
)

# 질문 실행
input_text = "삼성전자의 주요 사업영역은?"
chatbot_response = qa_chain.invoke(input_text)

print('내가 넣었던 질문:', chatbot_response['query'])
print('검색 결과를 바탕으로 GPT-4가 작성한 답변:', chatbot_response['result'])

print('챗봇이 참고한 실제 검색 결과에 해당하는 유사도 상위 3개 문서:')
for doc in chatbot_response['source_documents']:
    print(doc)
    print('--' * 100)