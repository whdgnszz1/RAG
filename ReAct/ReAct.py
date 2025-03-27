import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

load_dotenv()

urls = [
   "https://raw.githubusercontent.com/llama-index-tutorial/llama-index-tutorial/main/ch06/ict_japan_2024.pdf",
   "https://raw.githubusercontent.com/llama-index-tutorial/llama-index-tutorial/main/ch06/ict_usa_2024.pdf"
]

for url in urls:
   filename = url.split("/")[-1]
   response = requests.get(url)

   with open(filename, "wb") as f:
       f.write(response.content)
   print(f"{filename} 다운로드 완료")
   
   embd = OpenAIEmbeddings()
   
def create_pdf_retriever(
    pdf_path: str,  # PDF 파일 경로
    persist_directory: str,  # 벡터 스토어 저장 경로
    embedding_model: OpenAIEmbeddings,  # OpenAIEmbeddings 임베딩 모델
    chunk_size: int = 512,  # 청크 크기 기본값 512
    chunk_overlap: int = 0  # 청크 오버랩 크기 기본값 0
) -> Chroma.as_retriever:

    # PDF 파일로드. 페이지 별로 로드.
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    # 청킹. 길이를 주면, 해당 길이가 넘지 않도록 자르는 것.
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, # PDF의 각 페이지를 최대 길이 512가 넘지 않도록 잘게 분할.
        chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(data)

    # 벡터 DB에 적재
    vectorstore = Chroma.from_documents(
        persist_directory=persist_directory,
        documents=doc_splits,
        embedding=embedding_model,
    )

    # 벡터 DB를 retriever 객체로 반환.
    return vectorstore.as_retriever()
    
# 일본 ICT 정책 데이터베이스 생성
retriever_japan = create_pdf_retriever(
    pdf_path="ict_japan_2024.pdf",
    persist_directory="db_ict_policy_japan_2024",
    embedding_model=embd
)

# 미국 ICT 정책 데이터베이스 생성
retriever_usa = create_pdf_retriever(
    pdf_path="ict_usa_2024.pdf",
    persist_directory="db_ict_policy_usa_2024",
    embedding_model=embd
)

jp_engine = create_retriever_tool(
    retriever=retriever_japan,
    name="japan_ict",
    description="일본의 ICT 시장동향 정보를 제공합니다. 일본 ICT와 관련된 질문은 해당 도구를 사용하세요.",
)

usa_engine = create_retriever_tool(
    retriever=retriever_usa,
    name="usa_ict",
    description="미국의 ICT 시장동향 정보를 제공합니다. 미국 ICT와 관련된 질문은 해당 도구를 사용하세요.",
)

tools = [jp_engine, usa_engine]

prompt_react = hub.pull("hwchase17/react")
print(prompt_react.template)
print('--프롬프트 끝--')

template = '''다음 질문에 최선을 다해 답변하세요. 당신은 다음 도구들에 접근할 수 있습니다:

{tools}

다음 형식을 사용하세요:

Question: 답변해야 하는 입력 질문
Thought: 무엇을 할지 항상 생각하세요
Action: 취해야 할 행동, [{tool_names}] 중 하나여야 합니다. 리스트에 있는 도구 중 1개를 택하십시오.
Action Input: 행동에 대한 입력값
Observation: 행동의 결과
... (이 Thought/Action/Action Input/Observation의 과정이 N번 반복될 수 있습니다)
Thought: 이제 최종 답변을 알겠습니다
Final Answer: 원래 입력된 질문에 대한 최종 답변

## 추가적인 주의사항
- 반드시 [Thought -> Action -> Action Input format] 이 사이클의 순서를 준수하십시오. 항상 Action 전에는 Thought가 먼저 나와야 합니다.
- 최종 답변에는 최대한 많은 내용을 포함하십시오.
- 한 번의 검색으로 해결되지 않을 것 같다면 문제를 분할하여 푸는 것은 중요합니다.
- 정보가 취합되었다면 불필요하게 사이클을 반복하지 마십시오.
- 묻지 않은 정보를 찾으려고 도구를 사용하지 마십시오.

시작하세요!

Question: {input}
Thought: {agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# GPT-4o로부터 llm 객체를 선언
llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

# prompt_react를 사용하면 영어 프롬프트. prompt를 사용하면 한글 프롬프트
react_agent = create_react_agent(llm, tools=tools, prompt=prompt)

react_agent_executor = AgentExecutor(
    agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True
)

result = react_agent_executor.invoke({"input": "한국과 미국의 ICT 기관 협력 사례"})

print('최종 답변:', result['output'])

# 멀티 쿼리
result = react_agent_executor.invoke({"input": "미국과 일본의 ICT 주요 정책의 공통점과 차이점을 설명해줘."})

print('최종 답변:', result['output'])

result = react_agent_executor.invoke({"input": "미국의 ICT 관련 정부 기구, 주요 법령, 국내 기업 진출 사례 각각 따로 검색해. 그렇게 해서 정보 좀 모아봐. 그리고 나서 일본의 AI 정책도 알려줘."})

print('최종 답변:', result['output'])