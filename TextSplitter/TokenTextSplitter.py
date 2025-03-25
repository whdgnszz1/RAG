from transformers import GPT2TokenizerFast
from langchain_text_splitters import CharacterTextSplitter

with open("./data/appendix-keywords.txt") as f:
    file = f.read()  # 파일의 내용을 읽어서 file 변수에 저장
    
    
hf_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    # 허깅페이스 토크나이저를 사용하여 CharacterTextSplitter 객체를 생성
    hf_tokenizer,
    chunk_size=300,
    chunk_overlap=50,
)
# state_of_the_union 텍스트를 분할하여 texts 변수에 저장
texts = text_splitter.split_text(file)

print(texts[1])