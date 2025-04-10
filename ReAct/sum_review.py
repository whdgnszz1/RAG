import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_teddynote import logging
from konlpy.tag import Okt
from collections import Counter
import random
import numpy as np

# 환경 변수 로드 및 로깅 설정
load_dotenv()
logging.langsmith("review")

# CSV 파일 경로 설정
csv_file_path = os.path.join('data', 'reviews.csv')

# 불용어 파일 경로 설정
stop_words_file = os.path.join('data', 'stop_words.txt')

# 출력 디렉토리 설정
base_result_dir = 'result/recom_weight_kywr'
os.makedirs(base_result_dir, exist_ok=True)

# CSV 파일을 DataFrame으로 읽기
try:
    df = pd.read_csv(csv_file_path, low_memory=False)
except FileNotFoundError:
    print(f"{csv_file_path} 파일을 찾을 수 없습니다. '/data' 디렉토리에 파일을 확인해주세요.")
    exit()

# 불용어 로드
stop_words = set()
try:
    with open(stop_words_file, 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())
    print(f"불용어 파일 '{stop_words_file}'에서 {len(stop_words)}개의 불용어를 로드했습니다.")
except FileNotFoundError:
    print(f"불용어 파일 '{stop_words_file}'을 찾을 수 없습니다. 기본 불용어 없이 진행합니다.")
except Exception as e:
    print(f"불용어 파일 읽기 중 오류 발생: {e}. 기본 불용어 없이 진행합니다.")

# ChatOpenAI 모델 초기화
llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)


# 키워드 추출 함수 (불용어 제외)
def extract_keywords(reviews, top_n=10):
    okt = Okt()
    keywords = []
    for review in reviews:
        tokens = okt.pos(str(review), norm=True, stem=True)
        keywords.extend([word for word, pos in tokens if pos in ['Noun', 'Adjective'] and word not in stop_words])
    keyword_counts = Counter(keywords)
    return keyword_counts.most_common(top_n)


# 리뷰 요약 및 키워드 추출을 위한 프롬프트 템플릿
summary_prompt = PromptTemplate.from_template(
    '''다음은 특정 상품에 대한 리뷰 목록과 평점 정보입니다. 이 리뷰들을 읽고, 다음을 수행해 주세요:
    1. 자연스럽고 간결하게 요약: 긍정적인 의견, 부정적인 의견, 전반적인 감상을 포함하며, 평점 데이터(평균 평점 및 분포)를 반영하여 객관적인 평가를 추가.
    2. 리뷰에서 주요 키워드 5개를 추출: 명사와 형용사 중 빈도수가 높은 키워드를 선정하되, 일반적인 조사나 접속사(예: "을", "에", "하고")는 제외.

    리뷰 목록:
    {reviews}

    평점 정보:
    - 평균 평점: {average_rating:.2f}
    - 평점 분포: {rating_distribution}

    응답 형식:
    요약: [여기에 요약 내용 작성]
    키워드: [키워드1], [키워드2], [키워드3], [키워드4], [키워드5]
    '''
)

# 중간 요약 프롬프트
intermediate_summary_prompt = PromptTemplate.from_template(
    '''다음은 여러 리뷰 요약 결과입니다. 이를 종합하여 자연스럽고 간결한 중간 요약을 작성해 주세요. 긍정적인 의견, 부정적인 의견, 전반적인 감상을 포함하며, 평점 데이터를 반영해 주세요. 반드시 "중간 요약:"이라는 구분자를 포함한 형식으로 응답해 주세요.

    요약 결과:
    {summaries}

    평점 정보:
    - 평균 평점: {average_rating:.2f}
    - 평점 분포: {rating_distribution}

    응답 형식:
    중간 요약: [여기에 중간 요약 내용 작성]
    '''
)

# 최종 요약 프롬프트
final_summary_prompt = PromptTemplate.from_template(
    '''다음은 여러 중간 요약 결과입니다. 이를 종합하여 최종적으로 자연스럽고 간결한 요약을 작성해 주세요. 긍정적인 의견, 부정적인 의견, 전반적인 감상을 포함하며, 평점 데이터를 반영해 주세요. 반드시 "최종 요약:"이라는 구분자를 포함한 형식으로 응답해 주세요.

    중간 요약 결과:
    {intermediate_summaries}

    평점 정보:
    - 평균 평점: {average_rating:.2f}
    - 평점 분포: {rating_distribution}

    응답 형식:
    최종 요약: [여기에 최종 요약 내용 작성]
    '''
)

# 상품별로 그룹화
grouped = df.groupby('sale_cmdtid')

for sale_cmdtid, group in grouped:
    # 결과 디렉토리 생성
    result_dir = os.path.join(base_result_dir, sale_cmdtid)
    os.makedirs(result_dir, exist_ok=True)

    # 리뷰와 평점, 추천 수 추출
    group_data = group[['revw_cntt', 'revw_rvgr', 'revw_rcmn_cont']].dropna(subset=['revw_cntt'])
    reviews = group['revw_cntt'].dropna().tolist()
    ratings = group['revw_rvgr'].dropna()
    recommend_counts = group_data['revw_rcmn_cont'].fillna(0).astype(int)

    if not reviews:
        continue  # 리뷰가 없으면 건너뜀

    # 리뷰가 100개 이상이면 랜덤으로 100개 추출
    if len(reviews) > 100:
        reviews = random.sample(reviews, 100)

    # 평점 통계 계산
    average_rating = ratings.mean() if not ratings.empty else 0
    rating_distribution = ratings.value_counts().sort_index() if not ratings.empty else pd.Series()

    # 키워드 추출 (불용어 제외)
    top_keywords = extract_keywords(reviews, top_n=10)

    # 리뷰를 5개씩 청킹
    chunk_size = 5
    review_chunks = [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
    recommend_chunks = [recommend_counts[i:i + chunk_size].tolist() for i in
                        range(0, len(recommend_counts), chunk_size)]

    # 각 청크별 요약 및 키워드 생성, LLM 키워드 수집 (가중치 적용)
    summaries = []
    llm_keywords_weighted = Counter()  # 가중치가 적용된 키워드 카운터
    for idx, (chunk, rec_chunk) in enumerate(zip(review_chunks, recommend_chunks)):
        prompt_input = summary_prompt.format(
            reviews="\n".join(chunk),
            average_rating=average_rating,
            rating_distribution=rating_distribution.to_dict()
        )
        try:
            summary_response = llm.invoke(prompt_input)
            response_text = summary_response.content
            if "요약:" in response_text and "키워드:" in response_text:
                summary = response_text.split("요약:")[1].split("키워드:")[0].strip()
                keywords_raw = response_text.split("키워드:")[1].strip().split(", ")
                # 키워드에서 '['와 ']' 제거
                keywords = [kw.strip('[]') for kw in keywords_raw]
                summaries.append(summary)
                # 가중치 적용: 평균 추천 수가 0이면 정규분포에서 값 추출
                avg_recommend = sum(rec_chunk) / len(rec_chunk) if rec_chunk else 0
                if avg_recommend == 0:
                    avg_recommend = max(np.random.normal(loc=1, scale=0.5), 0.1)  # 평균 1, 표준편차 0.5, 최소 0.1
                for keyword in keywords:
                    llm_keywords_weighted[keyword] += avg_recommend
                    # 청크별 요약과 키워드 저장
                with open(os.path.join(result_dir, f'chunk_{idx}_summary.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"요약: {summary}\n키워드: {', '.join(keywords)}")
            else:
                print(f"[{sale_cmdtid}] 청크 {idx} 처리 중 예상치 못한 응답 형식: {response_text}")
                summaries.append("요약 생성 실패")
        except Exception as e:
            print(f"[{sale_cmdtid}] 청크 {idx} 처리 중 오류 발생: {e}")
            summaries.append(f"오류로 인한 요약 실패: {str(e)}")

    # 중간 요약 생성 (5개씩 청킹)
    intermediate_summaries = []
    intermediate_chunk_size = 5
    summary_chunks = [summaries[i:i + intermediate_chunk_size] for i in
                      range(0, len(summaries), intermediate_chunk_size)]

    for idx, chunk in enumerate(summary_chunks):
        prompt_input = intermediate_summary_prompt.format(
            summaries="\n\n".join(chunk),
            average_rating=average_rating,
            rating_distribution=rating_distribution.to_dict()
        )
        try:
            intermediate_response = llm.invoke(prompt_input)
            response_text = intermediate_response.content
            if "중간 요약:" in response_text:
                intermediate_summary = response_text.split("중간 요약:")[1].strip()
                intermediate_summaries.append(intermediate_summary)
                with open(os.path.join(result_dir, f'intermediate_{idx}_summary.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"중간 요약: {intermediate_summary}")
            else:
                print(f"[{sale_cmdtid}] 중간 요약 {idx} 처리 중 예상치 못한 응답 형식: {response_text}")
                intermediate_summaries.append("중간 요약 생성 실패")
        except Exception as e:
            print(f"[{sale_cmdtid}] 중간 요약 {idx} 처리 중 오류 발생: {e}")
            intermediate_summaries.append(f"오류로 인한 중간 요약 실패: {str(e)}")

    # 최종 요약 생성
    final_prompt_input = final_summary_prompt.format(
        intermediate_summaries="\n\n".join(intermediate_summaries),
        average_rating=average_rating,
        rating_distribution=rating_distribution.to_dict()
    )

    try:
        final_summary_response = llm.invoke(final_prompt_input)
        response_text = final_summary_response.content
        if "최종 요약:" in response_text:
            final_summary = response_text.split("최종 요약:")[1].strip()
        else:
            final_summary = "최종 요약 생성 실패: 예상치 못한 응답 형식"
    except Exception as e:
        print(f"[{sale_cmdtid}] 최종 요약 생성 중 오류 발생: {e}")
        final_summary = f"최종 요약 생성 실패: {str(e)}"

    # 결과 출력 및 파일 저장
    result_text = f"=== 상품 {sale_cmdtid}에 대한 리뷰 요약 ===\n"
    result_text += f"{final_summary}\n\n"
    result_text += f"=== 평점 통계 ===\n"
    result_text += f"평균 평점: {average_rating:.2f}\n"
    result_text += "평점 분포:\n"
    result_text += f"{rating_distribution.to_string()}\n\n"
    result_text += f"=== 상위 10개 키워드 (코드 계산) ===\n"
    for keyword, count in top_keywords:
        result_text += f"{keyword}: {count}회\n"
    result_text += f"\n=== 전체 키워드 (LLM 추출, 가중치 적용) ===\n"
    for keyword, weighted_count in sorted(llm_keywords_weighted.items(), key=lambda x: (-x[1], x[0])):
        result_text += f"{keyword}: {weighted_count:.2f} (가중치 합계)\n"

    print(result_text)

    # 최종 요약 파일 저장
    with open(os.path.join(result_dir, 'final_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(result_text)

    # LLM 키워드 별도 파일 저장
    with open(os.path.join(result_dir, 'llm_keywords.txt'), 'w', encoding='utf-8') as f:
        f.write(f"=== 전체 키워드 (LLM 추출, 가중치 적용) ===\n")
        for keyword, weighted_count in sorted(llm_keywords_weighted.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{keyword}: {weighted_count:.2f}\n")

print(f"모든 요약 결과가 '{base_result_dir}' 디렉토리에 저장되었습니다.")
