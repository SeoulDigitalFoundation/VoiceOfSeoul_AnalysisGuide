# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 00 라이브러리 로딩 #

# %%
import pandas as pd
import numpy as np
import regex
import gensim
from PyKomoran import *
komoran = Komoran("STABLE")
from tqdm import tqdm
tqdm.pandas()
import pickle

import seaborn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')

# %% [markdown]
# # 01 학습 데이터  #
# %% [markdown]
# ## 데이터 전처리 ##
# %% [markdown]
# ### 천만상상오아시스 파일 확인 ##

# %%
train_df = pd.read_csv('train.csv',sep='\t')
train_df

# %% [markdown]
# ### 컬럼 확인 ###

# %%
train_df.columns


# %%
# 민원 분류별 개수 #
train_df['분류'].value_counts()


# %%
# 연도별 민원 개수 #
train_df['작성일'] = pd.to_datetime(train_df['작성일'])
train_df['작성일'].dt.year.value_counts()

# %% [markdown]
# ### 데이터 핸들링 ###

# %%
# '내용' 없는 행 삭제 #
train_df = train_df.dropna(subset=['내용'])


# %%
# '내용' 중복 행 삭제 #
train_df = train_df.drop_duplicates(subset='내용')


# %%
# 전처리 후 민원 분류 개수 및 연도별 민원 개수 재확인 #
print(train_df['분류'].value_counts())
print(train_df['작성일'].dt.year.value_counts())


# %%
# 제목 + 내용 통합 #
train_df['contents'] = train_df.apply(lambda x:x['제목']+"\n"+x['내용'],axis=1)
train_df['contents']

# %% [markdown]
# ## 형태소 분석 ##
# %% [markdown]
# ### 사용자 사전 추가 ###

# %%
komoran.set_user_dic('dic.txt')


# %%
komoran.get_list('강변북로 교통혼잡')


# %%
#특수문자 제거#
def cleanText(readData):
    #텍스트에 포함되어 있는 특수 문자 제거
    #text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    text = re.sub(r'[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\’\“\”\·\n\r\t■◇◆▶;\xa0]', '', readData).strip()
    return text

#명사, 형용사, 관형사, 동사, 부사#
def morp(strings):
   return [w.get_first()+'다' if w.get_second() in ['VV','VA'] else w.get_first() for w in komoran.get_list(cleanText(strings)) if w.get_second() in ['NNP','NNG','MAG','VA','VV','MM']]


# %%
morp('혼잡스러운 강변북로를 달리다')

# %% [markdown]
# ### 토크나이징 ###

# %%
# 특정 품사 토크나이징
train_df['tokens'] = train_df['contents'].progress_map(lambda x:morp(x))


# %%
# 모든 품사 토크나이징
train_df['all_tokens'] = train_df['contents'].progress_map(lambda x:komoran.get_morphes_by_tags(cleanText(x)))

# %% [markdown]
# ### 불용어(stopwords) 제거 ###

# %%
# 불용어 리스트 만들기
stopwords = ['아래', '상상', '제안', '까지', '닷컴', '포털', '사이트', '천만', '오아시스', '이벤트', '접수','서울시','서울','특별시',
             '천만상상','파일','첨부','응모','슬로건','공모','공모전','응모전','신청','경우','때문','정도','사항',
                   '해당','겁니다','이것','저것','그것','돋움','신명', '태명', '한컴', '돋움',
                   '동안','거기','저기','여기','대부분','누구','무엇','고딕','만큼','굴림','감사','건지','텐데',
                   '안녕','안녕하세요','이번','걸로','수고','겁니까','그간','그건','그때','글쓴이','누가','니다','다면',
                   '뭔가','상상오아시스','하다','이다','되다','같다','궁','자체','서체','정','서','이','을','있다','없다', '체','관련',
                   '생각', '현재', '진행', '사람', '마음', '남산', '내용', '현실','음','막','김','변','조',
                   '오','참','동','지금','주변','대상','부분','요즘','하루','마련','세대','시간','이상','행위',
                   '활동','구분','사실','과정','모습','기간','선정','단지','자신','발생','지역','기대','마련',
                   '장소','모두','부탁','제공','이용','해주','당시','최근','민원','문제','문제점','현황','개선','방안',
                   '문의','답변','일동','요청','담당자','직원','방법','사용','활용','확인','방식','예전','안녕하십니까',
                   '이하','바로 가기','바로가기','제가']
                   
# 기존 불용어 리스트에 추가
stopwords.append('홈페이지')
stopwords_set = set(stopwords)


# %%
# 불용어 제거
train_df['tokens'] = train_df['tokens'].map(lambda x:[w for w in x if not w in stopwords_set])
train_df['all_tokens'] = train_df['all_tokens'].map(lambda x:[w for w in x if not w in stopwords_set])


# %%
train_df['tokens']


# %%
train_df


# %%
# 토크나이징 후에도 중복된 행 제거
train_df["res"] = train_df["tokens"].astype(str) #list to string
train_df = train_df.drop_duplicates(subset="res")
train_df = train_df.drop('res',1)
train_df = train_df.reset_index(drop=True) # index 리셋


# %%
train_df

# %% [markdown]
# # 02 테스트 데이터 #
# %% [markdown]
# ## 민주주의 서울 자유제안 ##
# ### 천만상상오아시스 학습데이터와 달리 민원 분류 컬럼이 없음 ###
# ### 학습데이터를 활용 해 테스트 데이터에 민원 분류를 자동으로 진행하고자 함 ###

# %%
# 민주주의 서울 제유제안 정보 CSV 파일의 경우 pandas에서 로드할때 깨지는 문제 발생. Excel 파일로 진행
test_df = pd.read_excel('민주주의 서울 자유제안 정보.xlsx')


# %%
test_df

# %% [markdown]
# ## 데이터 전처리 ##
# %% [markdown]
# ### 데이터 핸들링 ###

# %%
# '제안내용' 빈칸인 행 삭제
test_df = test_df.dropna(subset=['제안내용'])


# %%
# '제안내용' 중복 행 삭제
test_df = test_df.drop_duplicates(subset=['제안내용'])


# %%
# '제안제목' + '제안내용' = contents 컬럼
test_df['contents'] = test_df.apply(lambda x:x['제안제목']+'\n'+x['제안내용'],axis=1)
test_df['contents']


# %%
# contents 컬럼 중복 행 삭제
test_df = test_df.drop_duplicates('contents')


# %%
# '제안등록일자' 컬럼 시계열 데이터 타입으로 변경
test_df['제안등록일자'] = pd.to_datetime(test_df['제안등록일자'])
test_df['제안등록일자']

# %% [markdown]
# ### 텍스트 전처리 ###

# %%
#html tag 內 한글 제거
def remove_tag(content):
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner,'',content)
    return cleantext


# %%
# 태그 패턴
tag_pattern = regex.compile(r'class\=\"*\p{Hangul}+\"*|font-family\: \"*\p{Hangul}+\"*|HY중고딕|서울남산체|맑은 고딕|함초롬바탕|굴림|굴림체|새굴림|고딕|나눔고딕|산돌고딕|모던고딕|한양중고딕|HY견명조|바탕|태그래픽|궁서|궁서체|휴먼 명조|휴먼명조|\&nbsp\;')


# %%
# 여러 html 태그 패턴 삭제
test_df['contents'] = test_df['contents'].map(lambda x:regex.sub(tag_pattern,'',remove_tag(x)))


# %%
test_df['contents']

# %% [markdown]
# ## 형태소 분석 ##
# %% [markdown]
# ### 토크나이징 ###

# %%
# 특정 품사 토크나이징
test_df['tokens'] = test_df['contents'].progress_map(lambda x:morp(x))


# %%
# 모든 품사 토크나이징
test_df['all_tokens'] = test_df['contents'].progress_map(lambda x:komoran.get_morphes_by_tags(cleanText(x)))

# %% [markdown]
# ### 불용어(Stopwords) 제거 ###

# %%
test_df['tokens'] = test_df['tokens'].map(lambda x:[w for w in x if not w in stopwords_set])
test_df['all_tokens'] = test_df['all_tokens'].map(lambda x:[w for w in x if not w in stopwords_set])


# %%
# 토크나이징 후에도 중복된 행 제거
test_df["res"] = test_df["tokens"].astype(str) #list to string
test_df = test_df.drop_duplicates(subset="res")
test_df = test_df.drop('res',1)
test_df = test_df.reset_index(drop=True) # index 리셋


# %%
test_df

