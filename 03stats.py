# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 00 라이브러리 & 민원분류 결과 로딩
# %% [markdown]
# ### 라이브러리

# %%
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ### 민원분류 결과 불러오기

# %%
result_df = pd.read_excel('민원분류 예측결과.xlsx')
result_df

# %% [markdown]
# # 01 기술통계량 분석 #
# %% [markdown]
# ## 민원 분류 예측결과 테이블로 정리

# %%
category_table = result_df[['분류','분류2']].apply(pd.Series.value_counts)
category_table

# %% [markdown]
# ### '분류' 내림차순으로 재정렬

# %%
category_table = category_table.sort_values(by='분류',ascending=False)
category_table

# %% [markdown]
# ### 민원 분류 예측결과 테이블 엑셀로 저장

# %%
category_table.to_excel('./stats/민원분류 예측결과 개수.xlsx',index=None)

# %% [markdown]
# ## 연도별 민원 비율 트랜드 분석
# %% [markdown]
# ### '연도' 컬럼 생성

# %%
result_df['연도'] = result_df['제안등록일자'].dt.year
result_df['연도']

# %% [markdown]
# ### 민원분류-연도 피벗 테이블 생성

# %%
# 연도별 민원 분류 개수 테이블
category_year_size = pd.crosstab(result_df['분류'],result_df['연도'])
category_year_size


# %%
# 연도별 민원 분류 비율 테이블
category_year_percent = pd.crosstab(result_df['분류'],result_df['연도']).apply(lambda x:x/x.sum()*100,axis=0)
category_year_percent


# %%
# 연도별 민원 분류 개수/비율 테이블 통합
pd.concat([category_year_size,category_year_percent],axis=1)


# %%
# 연도별 민원 분류 개수/비율 테이블 엑셀로 저장
pd.concat([category_year_size,category_year_percent],axis=1).to_excel('./stats/연도별 민원분류 통계.xlsx',index=None)

# %% [markdown]
# # 02 민원 단어 빈도수 분석
# %% [markdown]
# ## 전체 민원 빈도수 분석
# %% [markdown]
# ### 단어 빈도 벡터 생성

# %%
tf_vectorizer = CountVectorizer(analyzer='word',
                             lowercase=False,
                             tokenizer=None,
                             preprocessor=None,
                             min_df=5,
                             ngram_range=(1,2)
                             )

# %% [markdown]
# ### 빈 벡터에 단어 리스트 투입

# %%
tf_vector = tf_vectorizer.fit_transform(result_df['tokens'].astype(str))

# %% [markdown]
# ### 단어 빈도수 계산

# %%
tf_scores = tf_vector.toarray().sum(axis=0)
tf_idx = np.argsort(-tf_scores)
tf_scores = tf_scores[tf_idx]
tf_vocab = np.array(tf_vectorizer.get_feature_names())[tf_idx]

# %% [markdown]
# ### 빈도수 상위 200개 단어 출력

# %%
print(list(zip(tf_vocab, tf_scores))[:200])

# %% [markdown]
# ### 단어 빈도수 테이블 생성(상위 200개)

# %%
tf_vocab_table = pd.DataFrame(list(zip(tf_vocab, tf_scores))[:200])
tf_vocab_table

# %% [markdown]
# ### 컬럼 이름 부여

# %%
tf_vocab_table.columns = ['단어','빈도수']
tf_vocab_table

# %% [markdown]
# ### 단어-빈도수 테이블 엑셀로 저장

# %%
tf_vocab_table.to_excel('./stats/전체_단어_빈도수.xlsx',index=None)

# %% [markdown]
# ## 주요 민원 분류 단어 빈도수 분석
# %% [markdown]
# ### '환경' 분류
# %% [markdown]
# ### '환경' 분류 토큰 컬럼

# %%
result_df[result_df['분류']=='환경']['tokens']

# %% [markdown]
# ### 빈 벡터에 '환경' 분류 단어 리스트 투입

# %%
tf_vector_environ = tf_vectorizer.fit_transform(result_df[result_df['분류']=='환경']['tokens'].astype(str))

# %% [markdown]
# ### 단어 빈도수 계산

# %%
tf_scores_environ = tf_vector_environ.toarray().sum(axis=0)
tf_idx_environ = np.argsort(-tf_scores_environ)
tf_scores_environ = tf_scores_environ[tf_idx_environ]
tf_vocab_environ = np.array(tf_vectorizer.get_feature_names())[tf_idx_environ]

# %% [markdown]
# ### '환경' 분류 빈도수 상위 200개 단어 출력

# %%
print(list(zip(tf_vocab_environ, tf_scores_environ))[:200])

# %% [markdown]
# ### 단어 빈도수 테이블 생성(상위 200개)

# %%
tf_vocab_environ_table = pd.DataFrame(list(zip(tf_vocab_environ, tf_scores_environ))[:200])
tf_vocab_environ_table.columns = ['단어','빈도수']
tf_vocab_environ_table

# %% [markdown]
# ### 단어-빈도수 테이블 엑셀로 저장

# %%
tf_vocab_environ_table.to_excel('./stats/환경_단어_빈도수.xlsx',index=None)

# %% [markdown]
# ### '교통' 분류
# %% [markdown]
# ### '교통' 분류 토큰 컬럼

# %%
result_df[result_df['분류']=='교통']['tokens']

# %% [markdown]
# ### 빈 벡터에 '교통' 분류 단어 리스트 투입

# %%
tf_vector_traffic = tf_vectorizer.fit_transform(result_df[result_df['분류']=='교통']['tokens'].astype(str))

# %% [markdown]
# ### 단어 빈도수 계산

# %%
tf_scores_traffic = tf_vector_traffic.toarray().sum(axis=0)
tf_idx_traffic = np.argsort(-tf_scores_traffic)
tf_scores_traffic = tf_scores_traffic[tf_idx_traffic]
tf_vocab_traffic = np.array(tf_vectorizer.get_feature_names())[tf_idx_traffic]

# %% [markdown]
# ### '교통' 분류 빈도수 상위 200개 단어 출력

# %%
print(list(zip(tf_vocab_traffic, tf_scores_traffic))[:200])

# %% [markdown]
# ### 단어 빈도수 테이블 생성(상위 200개) 

# %%
tf_vocab_traffic_table = pd.DataFrame(list(zip(tf_vocab_traffic, tf_scores_traffic))[:200])
tf_vocab_traffic_table.columns = ['단어','빈도수']
tf_vocab_traffic_table

# %% [markdown]
# ### 단어-빈도수 테이블 엑셀로 저장

# %%
tf_vocab_traffic_table.to_excel('./stats/교통_단어_빈도수.xlsx',index=None)

