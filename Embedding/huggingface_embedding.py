from sentence_transformers import SentenceTransformer
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd

model = SentenceTransformer('BAAI/bge-m3')

def get_embedding(text):
    return list(model.encode(text))
    
embedding_result = get_embedding('저는 배가 고파요')
print(embedding_result)

data = ['저는 배가 고파요',
        '저기 배가 지나가네요',
        '굶어서 허기가 지네요',
        '허기 워기라는 게임이 있는데 즐거워',
        '스팀에서 재밌는 거 해야지',
        '스팀에어프라이어로 연어구이 해먹을거야']

df = pd.DataFrame(data, columns=['text'])

df['embedding'] = df.apply(lambda row: get_embedding(
        row.text
    ), axis=1)
    
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    # query라고 하는 텍스트가 들어오면 get_embedding이라는 함수를 통해서 벡터값을 얻음.
    # query라고 하는 텍스트의 임베딩 값은 query_embedding에 저장이 됩니다.
    query_embedding = get_embedding(
        query
    )

    # query라는 텍스트가 임베딩이 된 query_embedding과
    # 데이터프레임 df의 embedding 열에 있는 모든 임베딩 벡터값들과 유사도를 계산을 하여
    # similarity 열에다가 각각의 유사도 점수를 기록.
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x),
                                                            np.array(query_embedding)))

    # similarity 열에 있는 유사도 값 기준으로 상위 3개의 행만 반환
    results_co = df.sort_values("similarity",
                                ascending=False,
                                ignore_index=True)
    return results_co.head(3)

sim_result = return_answer_candidate(df, '아무 것도 안 먹었더니 꼬르륵 소리가나네')
print(sim_result)