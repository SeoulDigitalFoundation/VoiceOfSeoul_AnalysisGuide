# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # 00 라이브러리 로딩 #

# %%
import pandas as pd
import numpy as np
import pickle

from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g

import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
plt.rc('font', family='Malgun Gothic')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# %%
# Load variables #
# 전처리 완료된 학습/테스트 데이터 바로 불러오기 #
with open('train_test_df.pickle', 'rb') as pic:
    train_df, test_df = pickle.load(pic)

# %% [markdown]
# # 01 기술통계량 #
# %% [markdown]
# ## 민원 건당 단어 수 ##
# ### Word2vec 모델 파라메터 값 설정에 참고될 통계량 ###
# %% [markdown]
# ### 학습 데이터 ###

# %%
train_df['tokens'].str.len().describe()

# %% [markdown]
# ### 테스트 데이터 ###

# %%
test_df['tokens'].str.len().describe()

# %% [markdown]
# # 02 Word2vec 모델링 #
# ### 코드 참고 : https://programmers.co.kr/learn/courses/21 ###
# %% [markdown]
# ## Word2vec 파라미터 값 지정 ##

# %%
train_documents = train_df['tokens'].to_list() # 학습데이터 토크나이징 완료한 문서
test_documents = test_df['tokens'].to_list() # 학습데이터 토크나이징 완료한 문서


# %%
num_features = 500 # 문자 벡터 차원 수
min_word_count = 40 # 최소 문자 수
num_workers = 6 # 병렬 처리 스레드 수
context = 20 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도수 Downsample

# %% [markdown]
# ## 모델 학습 ##

# %%
model = word2vec.Word2Vec(train_documents,
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)


# %%
# 모델 확인
model

# %% [markdown]
# ### 특정 단어와 가장 유사한 단어 추출 ###

# %%
model.wv.most_similar('자전거')


# %%
model.wv.most_similar('미세먼지')

# %% [markdown]
# ### 입력한 복수의 단어 中 유사하지 않은 단어 추출 ###

# %%
model.wv.doesnt_match('자전거 버스 택시'.split())


# %%
model.wv.doesnt_match('임산부 노인 시민'.split())

# %% [markdown]
# ### 확습 완료 후 필요없는 메모리 unload ###

# %%
model.init_sims(replace=True)

# %% [markdown]
# ### 모델 저장 ###

# %%
model_name = '500features_40minwords_20context' #모델 이름 지정
model.save(f'./model/{model_name}')

# %% [markdown]
# ### t-SNE로 word2vec 모델 시각화 ###

# %%
# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False
model = g.Doc2Vec.load(f'./model/{model_name}')

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2) # 2개의 차원


# %%
# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100,:])
# X_tsne = tsne.fit_transform(X)

w2v_df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
w2v_df.shape
w2v_df.head(10)

fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(w2v_df['x'], w2v_df['y'])

for word, pos in w2v_df.iterrows():
    ax.annotate(word, pos, fontsize=20)
plt.show()

# %% [markdown]
# ## 문서별 평균 feature 계산 ##
# %% [markdown]
# ### 주어진 민원 문서에서 단어 벡터의 평균 구하는 함수 ###

# %%
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    np.seterr(divide='ignore')
    featureVec = np.divide(featureVec,nwords)
    return featureVec

# %% [markdown]
# ### 단어 평균 feature 벡터를 배열로 반환 ###

# %%
def getAvgFeatureVecs(documents, model, num_features):
    counter = 0.
    documentFeatureVecs = np.zeros(
        (len(documents),num_features),dtype="float32")
    for document in documents:
       if counter%2000. == 0.:
           print("민원 %d of %d" % (counter, len(documents)))
       documentFeatureVecs[int(counter)] = makeFeatureVec(document, model, num_features)
       counter = counter + 1.
    return documentFeatureVecs

# %% [markdown]
# ### 학습 데이터 평균 feature 계산 ###

# %%
get_ipython().run_line_magic('time', 'trainDataVecs = getAvgFeatureVecs(    train_documents, model, num_features)')


# %%
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(trainDataVecs)
trainDataVecs=imp.transform(trainDataVecs) ##nan, infinite 제거

# %% [markdown]
# ## 랜덤 포레스트로 민원 자동분류 ##
# %% [markdown]
# ### 랜덤 포레스트 분류기 설정 ###

# %%
forest_w2v = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state=2019,class_weight='balanced')

# %% [markdown]
# ### 랜덤포레스트 모델 = Y: 민원분류, X:민원 문서당 평균 단어 feature ###

# %%
get_ipython().run_line_magic('time', 'forest_w2v = forest_w2v.fit(trainDataVecs, train_df["분류"])')

# %% [markdown]
# ### cross validation 활용해 정확도 평가 ###

# %%
# 10개로 분할해 평가
get_ipython().run_line_magic('time', 'score = np.mean(cross_val_score(    forest_w2v, trainDataVecs,     train_df["분류"], cv=10))')


# %%
# 정확도(accuracy)
score

# %% [markdown]
# ### 최종 모델을 활용해서 테스트 데이터 민원 자동 분류 ###

# %%
# 테스트 데이터 벡터화
get_ipython().run_line_magic('time', 'testDataVecs = getAvgFeatureVecs(        test_documents, model, num_features )')
imp.fit(testDataVecs)
testDataVecs=imp.transform(testDataVecs) ##nan, infinite 제거


# %%
# 문서별 민원 분류 확률값 저장
forest_w2v_proba = forest_w2v.predict_proba(testDataVecs)


# %%
# 민원 분류 확률 Top2
topn = 2
topn_class = np.argsort(forest_w2v_proba)[:,:-topn-1:-1]
forest_w2v.classes_[topn_class]

result_w2v_top1 = forest_w2v.classes_[topn_class][:,0]
result_w2v_top2 = forest_w2v.classes_[topn_class][:,1]


# %%
# 민원 분류 결과 데이터프레임으로 저장
output_w2v = pd.DataFrame(data={'제안번호':test_df['제안번호'], '분류':result_w2v_top1,'분류2':result_w2v_top2})
output_w2v


# %%
# 민원 예측 결과 통계량
print(output_w2v['분류'].value_counts())


# %%
# 민원 분류 예측 결과를 test_df에 merge
result_df=pd.merge(test_df,output_w2v,on='제안번호')
result_df

# %% [markdown]
# ### 분류 결과 도표 시각화 ###

# %%
#학습 데이터 비전 비율 vs 테스트 데이터 비전 비율
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)

#학습 데이터 플롯
train_plot = sns.countplot(train_df['분류'], ax=axes[0],order=train_df['분류'].value_counts().index.to_list())
train_plot.set_xticklabels(train_plot.get_xticklabels(),rotation=45,ha='right')
train_plot.set_title('학습 데이터')

#테스트 데이터 플롯
test_plot = sns.countplot(output_w2v['분류'], ax=axes[1],order=result_df['분류'].value_counts().index.to_list())
test_plot.set_xticklabels(test_plot.get_xticklabels(),rotation=45,ha='right')
test_plot.set_title('테스트 데이터')

