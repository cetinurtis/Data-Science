
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fancyimpute import SoftImpute, BiScaler

#Community Health Status Indicators (CHSI) to Combat Obesity, Heart Disease and Cancer
#https://healthdata.gov/dataset/community-health-status-indicators-chsi-combat-obesity-heart-disease-and-cancer
df0 = pd.read_csv("/Users/curtis/Desktop/Data Science/Medical/data/SUMMARYMEASURESOFHEALTH.csv")

#some columns are relevant for our purpose, these are mostly constant variables, most of them are not numeric
drop_columns=['State_FIPS_Code', 'County_FIPS_Code', 'CHSI_County_Name',
       'CHSI_State_Name', 'CHSI_State_Abbr', 'Strata_ID_Number','US_ALE', 'US_All_Death', 'US_Health_Status', 'US_Unhealthy_Days']

df=df0.drop(columns=drop_columns)

#the following are coded for missing values
df=df.replace(-1111.1 , np.nan, regex=True)
df=df.replace(-2222.2 , np.nan, regex=True)

df_original=df.copy()

#df has some nan cells, we want to put more nans RANDOMLY (this is important) so that the total nans will be %80 of the data.
nan_mat = np.random.random(df.shape)<0.788
df=df.mask(nan_mat)

#now there might be nan rows and columns
rows_nan = df.index[df.isnull().all(1)]
columns_nan = df.columns[df.isnull().all(0)]

df=df.drop(rows_nan)
df=df.drop(columns=columns_nan)
df_original=df_original.drop(rows_nan)
df_original=df_original.drop(columns=columns_nan)

#df is ready to impute

#we need to normalize data before using SoftImpute. There is biscalar build in SoftImpute which is not useful for this data
normalizer = StandardScaler()
X_norm = normalizer.fit_transform(df)

imputed = SoftImpute(max_iters=100).fit_transform(X_norm)
#the data has only 18 columns. For data with more columns the rank plays a role. This is just a demo.

#Go back to original scaling so we can compare the original values and find the error
X_unnorm = normalizer.inverse_transform(imputed)
df_imputed = pd.DataFrame(X_unnorm,columns = df.columns, index=df.index)

df_delta=df_original-df_imputed
#this is root-mean-square error (RMSE)
error=((df_delta)**2).mean()**0.5
print(error)

print(df_original.std())




[SoftImpute] Iter 100: observed MAE=0.027055 rank=18
[SoftImpute] Stopped after iteration 100 for lambda=0.873495
ALE                        1.465057
Min_ALE                    1.310512
Max_ALE                    1.036899
All_Death                 95.662086
Min_All_Death             65.326000
Max_All_Death             65.839865
CI_Min_All_Death          95.762378
CI_Max_All_Death         102.371257
Health_Status              4.565602
Min_Health_Status          2.778740
Max_Health_Status          4.536208
CI_Min_Health_Status       4.046970
CI_Max_Health_Status       5.524324
Unhealthy_Days             0.978868
Min_Unhealthy_Days         0.683101
Max_Unhealthy_Days         0.881938
CI_Min_Unhealthy_Days      1.077197
CI_Max_Unhealthy_Days      1.245796
dtype: float64

#df_original.std()
ALE                        1.993931
Min_ALE                    1.740446
Max_ALE                    1.495253
All_Death                130.920248
Min_All_Death             93.469510
Max_All_Death             93.556352
CI_Min_All_Death         128.533046
CI_Max_All_Death         143.879355
Health_Status              6.091730
Min_Health_Status          3.750335
Max_Health_Status          6.047315
CI_Min_Health_Status       5.400735
CI_Max_Health_Status       7.491431
Unhealthy_Days             1.342887
Min_Unhealthy_Days         0.829196
Max_Unhealthy_Days         1.194216
CI_Min_Unhealthy_Days      1.317164
CI_Max_Unhealthy_Days      1.671045
dtype: float64


# Remark: Mean Absolute Error: MAE=0.027055.  Root Square Mean Error (RSME) is calculated for each column. 
# I compared these with standard deviations of the original data. RMSE's are less than the standard deviations, which is good. 
# This is a demo I don't go further investigation.   
