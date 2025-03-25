from langchain_text_splitters import RecursiveCharacterTextSplitter
# 인기가 많음!
# 이 분할기는 청크가 충분히 작아질 때까지 주어진 문자 목록의 순서대로 텍스트를 분할하려고 시도
# 단락 -> 문장 -> 단어 순서로 재귀적으로 분할
# 1. 텍스트가 분할되는 방식: 문자 목록(`["\n\n", "\n", " ", ""]`) 에 의해 분할
# 2. 청크 크기가 측정되는 방식: 문자 수에 의해 측정

with open("./data/appendix-keywords.txt") as f:
    file = f.read()
    
text_splitter = RecursiveCharacterTextSplitter(
    # 청크 크기를 매우 작게 설정. 예시를 위한 설정.
    chunk_size=250,
    # 청크 간의 중복되는 문자 수를 설정
    chunk_overlap=50,
    # 문자열 길이를 계산하는 함수를 지정
    length_function=len,
    # 구분자로 정규식을 사용할지 여부를 설정
    is_separator_regex=False,
)

# text_splitter를 사용하여 file 텍스트를 문서로 분할합
texts = text_splitter.create_documents([file])
print(texts[0])  # 분할된 문서의 첫 번째 문서를 출력
print("===" * 20)
print(texts[1])  # 분할된 문서의 두 번째 문서를 출력

# 텍스트를 분할하고 분할된 텍스트의 처음 2개 요소를 반환
print(text_splitter.split_text(file)[:2])