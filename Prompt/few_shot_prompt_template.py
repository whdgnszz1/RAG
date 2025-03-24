from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import ChatOpenAI
from langchain_teddynote.messages import stream_response
from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_teddynote.prompts import CustomExampleSelector

# 환경 변수 로드 및 LangSmith 로깅 설정
load_dotenv()
logging.langsmith("default")

# ChatOpenAI 객체 생성
llm = ChatOpenAI(
    temperature=0,  # 창의성
    model_name="gpt-4-turbo",  # 모델명
)

# 예시 데이터
examples = [
    {
        "instruction": "당신은 회의록 작성 전문가 입니다. 주어진 정보를 바탕으로 회의록을 작성해 주세요",
        "input": "2023년 12월 25일, XYZ 회사의 마케팅 전략 회의가 오후 3시에 시작되었다. 회의에는 마케팅 팀장인 김수진, 디지털 마케팅 담당자인 박지민, 소셜 미디어 관리자인 이준호가 참석했다. 회의의 주요 목적은 2024년 상반기 마케팅 전략을 수립하고, 새로운 소셜 미디어 캠페인에 대한 아이디어를 논의하는 것이었다. 팀장인 김수진은 최근 시장 동향에 대한 간략한 개요를 제공했으며, 이어서 각 팀원이 자신의 분야에서의 전략적 아이디어를 발표했다.",
        "answer": """
회의록: XYZ 회사 마케팅 전략 회의
일시: 2023년 12월 25일
장소: XYZ 회사 회의실
참석자: 김수진 (마케팅 팀장), 박지민 (디지털 마케팅 담당자), 이준호 (소셜 미디어 관리자)

1. 개회
   - 회의는 김수진 팀장의 개회사로 시작됨.
   - 회의의 목적은 2024년 상반기 마케팅 전략 수립 및 새로운 소셜 미디어 캠페인 아이디어 논의.

2. 시장 동향 개요 (김수진)
   - 김수진 팀장은 최근 시장 동향에 대한 분석을 제시.
   - 소비자 행동 변화와 경쟁사 전략에 대한 통찰 공유.

3. 디지털 마케팅 전략 (박지민)
   - 박지민은 디지털 마케팅 전략에 대해 발표.
   - 온라인 광고와 SEO 최적화 방안에 중점을 둠.

4. 소셜 미디어 캠페인 (이준호)
   - 이준호는 새로운 소셜 미디어 캠페인에 대한 아이디어를 제안.
   - 인플루언서 마케팅과 콘텐츠 전략에 대한 계획을 설명함.

5. 종합 논의
   - 팀원들 간의 아이디어 공유 및 토론.
   - 각 전략에 대한 예산 및 자원 배분에 대해 논의.

6. 마무리
   - 다음 회의 날짜 및 시간 확정.
   - 회의록 정리 및 배포는 박지민 담당.
""",
    },
    {
        "instruction": "당신은 요약 전문가 입니다. 다음 주어진 정보를 바탕으로 내용을 요약해 주세요",
        "input": "이 문서는 '지속 가능한 도시 개발을 위한 전략'에 대한 20페이지 분량의 보고서입니다. 보고서는 지속 가능한 도시 개발의 중요성, 현재 도시화의 문제점, 그리고 도시 개발을 지속 가능하게 만들기 위한 다양한 전략을 포괄적으로 다루고 있습니다. 이 보고서는 또한 성공적인 지속 가능한 도시 개발 사례를 여러 국가에서 소개하고, 이러한 사례들을 통해 얻은 교훈을 요약하고 있습니다.",
        "answer": """
문서 요약: 지속 가능한 도시 개발을 위한 전략 보고서

- 중요성: 지속 가능한 도시 개발이 필수적인 이유와 그에 따른 사회적, 경제적, 환경적 이익을 강조.
- 현 문제점: 현재의 도시화 과정에서 발생하는 주요 문제점들, 예를 들어 환경 오염, 자원 고갈, 불평등 증가 등을 분석.
- 전략: 지속 가능한 도시 개발을 달성하기 위한 다양한 전략 제시. 이에는 친환경 건축, 대중교통 개선, 에너지 효율성 증대, 지역사회 참여 강화 등이 포함됨.
- 사례 연구: 전 세계 여러 도시의 성공적인 지속 가능한 개발 사례를 소개. 예를 들어, 덴마크의 코펜하겐, 일본의 요코하마 등의 사례를 통해 실현 가능한 전략들을 설명.
- 교훈: 이러한 사례들에서 얻은 주요 교훈을 요약. 강조된 교훈에는 다각적 접근의 중요성, 지역사회와의 협력, 장기적 계획의 필요성 등이 포함됨.
""",
    },
    {
        "instruction": "당신은 문장 교정 전문가 입니다. 다음 주어진 문장을 교정해 주세요",
        "input": "우리 회사는 새로운 마케팅 전략을 도입하려고 한다. 이를 통해 고객과의 소통이 더 효과적이 될 것이다.",
        "answer": "본 회사는 새로운 마케팅 전략을 도입함으로써, 고객과의 소통을 보다 효과적으로 개선할 수 있을 것으로 기대된다.",
    },
]

# Vector DB 생성
chroma = Chroma("fewshot_chat", OpenAIEmbeddings())

# 채팅 기반 예제 프롬프트 설정
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{instruction}:\n{input}"),
        ("ai", "{answer}"),
    ]
)

# 의미적 유사성 기반 예제 선택기
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    chroma,
    k=1,
)

# FewShotChatMessagePromptTemplate 설정
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
)

# 커스텀 예제 선택기 설정
custom_selector = CustomExampleSelector(examples, OpenAIEmbeddings())

custom_fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=custom_selector,
    example_prompt=example_prompt,
)

# 최종 커스텀 프롬프트 설정
custom_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        custom_fewshot_prompt,
        ("human", "{instruction}\n{input}"),
    ]
)

# 체인 생성
chain = custom_prompt | llm

# 테스트 입력
question = {
    "instruction": "회의록을 작성해 주세요",
    "input": "2023년 12월 26일, ABC 기술 회사의 제품 개발 팀은 새로운 모바일 애플리케이션 프로젝트에 대한 주간 진행 상황 회의를 가졌다. 이 회의에는 프로젝트 매니저인 최현수, 주요 개발자인 황지연, UI/UX 디자이너인 김태영이 참석했다. 회의의 주요 목적은 프로젝트의 현재 진행 상황을 검토하고, 다가오는 마일스톤에 대한 계획을 수립하는 것이었다. 각 팀원은 자신의 작업 영역에 대한 업데이트를 제공했고, 팀은 다음 주까지의 목표를 설정했다.",
}

# 결과 출력
print(stream_response(chain.stream(question)))
