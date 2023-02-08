# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:36:42 2023

@author: lsy
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

pd.set_option('display.max_columns', 3000)
pd.set_option('display.max_row', 3000)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.head()
test.head()

trainD = train.describe().transpose()
testD = test.describe().transpose()

# null plot
# 전체 칼럼을 보기에는 많이 어려움
# 그러므로 x는 500개씩 나눌 것임 => 6개의 plot이 나올 예정
dfNull = pd.DataFrame([train.columns, train.isnull().sum()]).transpose()
dfNull.columns = ["column", "null"]


testD.to_csv("기초통계량_test.csv")


length1 = 0
length2 = 500
for _ in range(6):
    if length1 == 2500:
        length2 = len(dfNull)
    
    f, ax = plt.subplots(figsize = (50,100))
    sns.set_theme(style = "whitegrid")
    sns.set_color_codes("pastel")
    sns.barplot(x = "null", y="column", data = dfNull[length1:length2], label = "null", color = "b")
    
    ax.set(xlabel = "null 개수", xlim = (0,598))
    sns.despine(left= True, bottom = True)
    plt.plot()
    
    length1 = length2
    length2 = length1 + 500

# 뭐 암튼 null값이 더럽게 많다 이말이야

# 단순 실수인 값들이 
# box plot
# 이번에 볼 것은 X 데이터의 분포임
# 기초통계량을 봤을 때 하나의 값만 갖고 나머지는 null을 가지고 있는 변수가 있음 이런 변수들은 선택지일수 있을 것 같음
# sns를 이용하여 box plot을 50행 10열 이런 형식으로 시각화를 해 보자


train.iloc[:,50]

# column 50개씩 잘라냈음
dfList = list()
for i in range(int(len(train.columns)/100)):
    dfList.append(train.iloc[:,i*100:(i+1)*100])
    
for i in range(len(dfList)):
    plt.figure(figsize = (10,30)) # 가로 세로
    dfList[i].boxplot(vert = False)

std0tr = trainD.loc[trainD["std"]==0]
std0te = testD.loc[testD["std"]==0]

std0tr = std0tr.reset_index() 
std0te = std0te.reset_index()

inter = set(std0tr["index"]).intersection(set(std0te["index"])) # 모두 겹치지는 않음 
outerTr = set(std0tr["index"]).difference(set(std0te["index"]))
outerTe = set(std0te["index"]).difference(set(std0tr["index"]))
outer = list(outerTr) + list(outerTe)

trainD = trainD.reset_index()
testD = testD.reset_index()

trOuter = train.describe()[outer].transpose()
trOuter.to_csv("outer기초통계량.csv")
teOuter = test.describe()[outer].transpose()
teOuter.to_csv("outer기초통계량.csv")


# outer의 기초통계량이 아닌 value count를 봐야할 것 같음
plt.figure(figsize = (50,50))
train[outer].plot.hist(subplots=True, figsize = (10,50))
train[outer].value_counts()


outTrCol = list()
outTeCol = list()
for col in outer:
    countTr = train[col].value_counts()
    countTe = test[col].value_counts()
    
    if len(countTr)>1:    
        outTrCol.append(col)        
    if len(countTe)>1:
        outTeCol.append(col)

plt.figure(figsize=(20,10))
for i in range(len(outTeCol)):
    plt.subplot(5,11,i+1)
    plt.tick_params(axis='x', which = 'both', bottom=False, top=False, labelbottom=False)
    plt.ylim(0,10)
    sns.countplot(test[outTeCol[i]])
plt.show()

