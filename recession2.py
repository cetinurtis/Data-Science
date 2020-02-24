import pandas as pd
from sympy import Interval, Union
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit

#Treasurer bills data
df = pd.read_csv("/Users/curtis/Desktop/stockfiles/Recession/GS10-TB3MS.csv", index_col=False)
#other parameters data
xls_file = pd.ExcelFile('/Users/curtis/Desktop/stockfiles/Recession/DataOrg.xlsx')
dfx=xls_file.parse('RInd')
#recession data from NBER
dfr = pd.read_csv("/Users/curtis/Desktop/stockfiles/Recession/NBER-1950.csv", index_col=False)

# construct recession intervals as one set
n = len(dfr['Start'])
k = Interval(0,0)
i=0
while i < n :
    k = Union(k,  Interval(dfr.iloc[i].values[0],dfr.iloc[i].values[1]))
    i += 1
# k is the set of recession intervals


#construct a column for the target variable: If there is an recession in 12 months it will 1, otherwise 0.
df['Month'] = df.Month.astype(float)
df['Target'] =[1 if x+12 in k else 0 for x in df['Month']]

m=len(df["Target"])
df.loc[m,"Target"]=0
dfx["Target"]=df["Target"]
#we focus on last 420 rows and last 5 columns (parameters)
dfx=dfx.iloc[-420:,-5:]
dfx = dfx.reset_index(drop=True)

#drop rows with nans
rows_nan = dfx.index[dfx.isnull().any(1)]
dfx_dropped= dfx.drop(rows_nan)
dfx_dropped = dfx_dropped.reset_index(drop=True)

#this is the target column
y=dfx_dropped["Target"]

X = dfx_dropped.drop(columns=["Target"])
#before applying the model we need to add a column of 1's
X = sm.add_constant(X)

#apply probit and logit regression models
#model = Probit(y, X.astype(float))
model = Logit(y, X.astype(float))
reg_model = model.fit()
print(reg_model.summary())

