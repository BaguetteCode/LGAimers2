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

train["X_770"].value_counts()
test["X_1811"].value_counts()

# nul
df = train.copy()
df.columns

# -1는 null을 의미함
df = df.fillna(-1)


for col in df.columns[6:]:
    df[col].loc[df[col].notnull()] = 0
    df[col].loc[df[col].isnull()] = 1

sns.countplot(data = df, x = "Y_Class", hue = "X_1")

fig, axis  = plt.subplots(10,10, figsize=(50,50))
fig.subplots_adjust(wspace=0.7, hspace=0.7)

columns = df.columns[6:]

def X_Y_null_plot(df, columns):
    fig, axis  = plt.subplots(10,10, figsize=(50,50))
    fig.subplots_adjust(wspace=0.7, hspace=0.7)

    k =0
    for i in range(10):
        for j in range(10):
            sns.scatterplot(data=df, x="Y_Class", y = columns[k])
            , ax=axis[i, j])
            k += 1
    plt.show()

col = list()
for i in range(29):
    col.append(columns[i*100:(i+1)*100])

for c in col:
    X_Y_null_plot(df, c)


상관관계 plot 
imputation 중 MICE 
Explainable ML의 관점

# df = pd.concat([train.iloc[:,6:], test.iloc[:,4:]], axis=1)

df1 = pd.DataFrame(list(df[df["Y_Class"]==0].sum(axis=0))[6:], columns = ["0"], index = df.columns[6:])
df1["1"] = list(df[df["Y_Class"]==1].sum(axis=0))[6:]
df1["2"] = list(df[df["Y_Class"]==2].sum(axis=0))[6:]
# df1["total"] = df1.sum(axis=1)
df1 = df1.transpose()
df1.to_csv("notNull count by y_class.csv", index = True)

dfList = df1.values.tolist()
dfListCol = df1.columns.tolist()
dfListIndex = df1.index.tolist()

dfDict = {k:v for k,v in zip(dfListIndex, dfList)}
dfDictL = {str(v):k for k,v in zip(dfListIndex, dfList)}

dfDictL = dict()
for k,v in zip(dfListIndex, dfList):
    try:
        dfDictL[str(v)].append(k)
    except:
        dfDictL[str(v)] = [k]

temp = [key[1:-1].replace(',', '').split(' ') for key in dfDictL.keys()]
temp2 = list()
for t in temp:
    temp2.append([float(t[0]), float(t[1]), float(t[2])])
    

gif = pd.DataFrame(temp2, columns = ["0","1","2"])
gif.plot.barh()

fig, ax = plt.subplots(3,1, figsize=(30,10))

gif = gif.transpose()
gif.columns = [str(i) for i in range(30)]
gif.to_csv("asdf.csv")


sns.set_palette("ch:start=.2,rot=-.3")
gif.iloc[:10].plot.bar()
gif.iloc[10:20].plot.bar()
gif.iloc[20:30].plot.bar()
plt.show()

