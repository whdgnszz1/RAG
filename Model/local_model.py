from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import GPT4All
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler

local_path = "../models/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf"

# 프롬프트
prompt = ChatPromptTemplate.from_template(
    """<s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.</s>
<s>Human: {question}</s>
<s>Assistant:
"""
)

# GPT4All 언어 모델 초기화
# model는 GPT4All 모델 파일의 경로를 지정
llm = GPT4All(
    model=local_path,
    backend="gpu",  # GPU 설정
    streaming=True,  # 스트리밍 설정
    callbacks=[StreamingStdOutCallbackHandler()],  # 콜백 설정
)

# 체인 생성
chain = prompt | llm | StrOutputParser()

# 질의 실행
response = chain.invoke({"question": "대한민국의 수도는 어디인가요?"})
