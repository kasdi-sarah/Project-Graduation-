import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.neural_network import MLPRegressor
#from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.dates as mdates
import sys
import numpy
import datetime
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
data = pd.read_excel('data.xlsx')      
data["Date"] = data["Date"].map(dt.datetime.toordinal)
df = data.copy()    

print("valeurs manquantes avant la supression")
print(df.isnull().sum())
print("valeurs manquantes apres la supression")
df=df.dropna(axis=0)
print(df.isnull().sum())

plt.figure()
sns.boxplot(x=df["Date"])
plt.figure()
sns.boxplot(x=df["pmErabRelAbnormalEnbAct"])
val_abr=['Date','pmErabRelAbnormalEnbAct']
# on a fait une boucle pour detecter les valeurs aberrantes 
for x in val_abr :
    q75,q25= np.percentile(df.loc[:,x],[75,25])
    intr_qr=q75-q25
    max=q75+(1.5*intr_qr)
    min=q25-(1.5*intr_qr)
    df.loc[df[x]<min,x]=np.nan
    df.loc[df[x]>max,x]=np.nan
# afficher le nombres des val abr  
print("valeurs aberrantes avant la supression")

print(df.isnull().sum())
print("valeurs aberrantes apres la supression")

df=df.dropna(axis=0)
# verifier si les val abr ont été suprrimé  
print(df.isnull().sum())
# x=np.array(df["Date"])
# y=np.array(df["pmErabRelAbnormalEnbAct"])
# plt.scatter(x , y, c ="blue")
# plt.xlim([737881, 738032])
# plt.ylim([-10, 400])
#filtré notre dataset on prenant que les colonne qui importe
df = df.drop(['Site_ID','Sector','Band','eNodeB_Drop [%]','pmErabRelAbnormalEnb',
              'pmErabRelNormalEnb','pmErabRelMme'], axis = 1)
def preprocess_inputs(df1):
    df1 = df1.copy()    
    X = df1.drop('pmErabRelAbnormalEnbAct', axis = 1)
    Y = df1['pmErabRelAbnormalEnbAct']
    plt.show()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, shuffle = True, random_state = 1)
    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = preprocess_inputs(df)
#------------------------------------------------lasso---------------------------------------
# #-------------------------------------------------------------------------------------------------
# lasso= Lasso()
# lasso.fit(X_train,Y_train)
# ypredtrain=lasso.predict(X_train)
# ypredtest=lasso.predict(X_test)
# MSEtrain = mean_squared_error(Y_train, ypredtrain)
# MSEtest = mean_squared_error(Y_test, ypredtest)
# print("la valeur de r2 d'entrainement pour lasso")
# r2_train=1 - (np.sum((Y_train - ypredtrain)**2) / np.sum((Y_train - Y_train.mean())**2))
# print(r2_train)
# print("la valeur de r2 de test  pour lasso ")
# r2_test=1 - (np.sum((Y_test - ypredtest)**2) / np.sum((Y_test - Y_test.mean())**2))
# print(r2_test)
# print("la valeur de rmse d'entrainement pour lasso")
# RMSEtrain = np.sqrt(MSEtrain)
# print(RMSEtrain)
# print("la valeur de rmse de test pour lasso")
# RMSEtest = np.sqrt(MSEtest)
# print(RMSEtest)
# MAEtrain =mean_absolute_error(Y_train,ypredtrain)
# MAEtest = mean_absolute_error(Y_test, ypredtest)
# print('mae lasso train',MAEtrain)
# print('mae lasso test',MAEtest)
# #---------------ridge---------------------
# ridgeReg = Ridge(alpha=10)
# ridgeReg.fit(X_train, Y_train)
# ypredtest=ridgeReg.predict(X_test)
# ypredtrain=ridgeReg.predict(X_train)
# MSEtrain = mean_squared_error(Y_train, ypredtrain)
# MSEtest = mean_squared_error(Y_test, ypredtest)
# print("la valeur de r2 d'entrainement pour ridge")
# r2_train=1 - (np.sum((Y_train - ypredtrain)**2) / np.sum((Y_train - Y_train.mean())**2))
# print(r2_train)
# print("la valeur de r2 de test  pour ridge ")
# r2_test=1 - (np.sum((Y_test - ypredtest)**2) / np.sum((Y_test - Y_test.mean())**2))
# print(r2_test)
# print("la valeur de rmse d'entrainement pour ridge")
# RMSEtrain = np.sqrt(MSEtrain)
# print(RMSEtrain)
# print("la valeur de rmse de test pour ridge")
# RMSEtest = np.sqrt(MSEtest)
# print(RMSEtest)
# MAEtrain =mean_absolute_error(Y_train,ypredtrain)
# MAEtest = mean_absolute_error(Y_test, ypredtest)
# print('mae ridge train',MAEtrain)
# print('mae ridge test',MAEtest)
# # # # #-------------------------------------------decision tree------------------------------
# from sklearn.tree import DecisionTreeRegressor
# arbre= DecisionTreeRegressor()
# arbre.fit(X_train,Y_train)
# ypredtrain=arbre.predict(X_train)
# ypredtest=arbre.predict(X_test)
# MSEtrain = mean_squared_error(Y_train, ypredtrain)
# MSEtest = mean_squared_error(Y_test, ypredtest)
# print("la valeur de r2 d'entrainement pour decision tree")
# r2_train=1 - (np.sum((Y_train - ypredtrain)**2) / np.sum((Y_train - Y_train.mean())**2))
# print(r2_train)
# print("la valeur de r2 de test  pour decision tree ")
# r2_test=1 - (np.sum((Y_test - ypredtest)**2) / np.sum((Y_test - Y_test.mean())**2))
# print(r2_test)
# print("la valeur de rmse d'entrainement pour decision tree")
# RMSEtrain = np.sqrt(MSEtrain)
# print(RMSEtrain)
# print("la valeur de rmse de test pour decision tree")
# RMSEtest = np.sqrt(MSEtest)
# print(RMSEtest)
# MAEtrain =mean_absolute_error(Y_train,ypredtrain)
# MAEtest = mean_absolute_error(Y_test, ypredtest)
# print('mae decision tree train',MAEtrain)
# print('mae decision tree test',MAEtest)

# # # ------------------------slr---------------------------------
# slr = linear_model.LinearRegression()
# slr.fit(X_train,Y_train)
# ypredtrain=slr.predict(X_train)
# ypredtest=slr.predict(X_test)
# MSEtrain = mean_squared_error(Y_train, ypredtrain)
# MSEtest = mean_squared_error(Y_test, ypredtest)
# print("la valeur de r2 d'entrainement pour slr")
# r2_train=1 - (np.sum((Y_train - ypredtrain)**2) / np.sum((Y_train - Y_train.mean())**2))
# print(r2_train)
# print("la valeur de r2 de test  pour slr ")
# r2_test=1 - (np.sum((Y_test - ypredtest)**2) / np.sum((Y_test - Y_test.mean())**2))
# print(r2_test)
# print("la valeur de rmse d'entrainement pour slr")
# RMSEtrain = np.sqrt(MSEtrain)
# print(RMSEtrain)
# print("la valeur de rmse de test pour slr")
# RMSEtest = np.sqrt(MSEtest)
# print(RMSEtest)
# MAEtrain =mean_absolute_error(Y_train,ypredtrain)
# MAEtest = mean_absolute_error(Y_test, ypredtest)
# print('mae slr train',MAEtrain)
# print('mae slr test',MAEtest)
# # ------------------------random forest---------------------------------
# rf = RandomForestRegressor(n_estimators=100)
# rf.fit(X_train,Y_train)
# ypredtrain=rf.predict(X_train)
# ypredtest=rf.predict(X_test)
# MSEtrain = mean_squared_error(Y_train, ypredtrain)
# MSEtest = mean_squared_error(Y_test, ypredtest)
# print("la valeur de r2 d'entrainement pour rf")
# r2_train=1 - (np.sum((Y_train - ypredtrain)**2) / np.sum((Y_train - Y_train.mean())**2))
# print(r2_train)
# print("la valeur de r2 de test  pour rf ")
# r2_test=1 - (np.sum((Y_test - ypredtest)**2) / np.sum((Y_test - Y_test.mean())**2))
# print(r2_test)
# print("la valeur de rmse d'entrainement pour rf")
# RMSEtrain = np.sqrt(MSEtrain)
# print(RMSEtrain)
# print("la valeur de rmse de test pour rf")
# RMSEtest = np.sqrt(MSEtest)
# print(RMSEtest)
# MAEtrain =mean_absolute_error(Y_train,ypredtrain)
# MAEtest = mean_absolute_error(Y_test, ypredtest)
# print('mae rf train',MAEtrain)
# print('mae rf test',MAEtest)
# # # # ---------------------svr-----------------------
# svr= SVR(C=1.0, epsilon=0.1)
# svr.fit(X_train,Y_train)
# ypredtrain=svr.predict(X_train)
# ypredtest=svr.predict(X_test)
# MSEtrain = mean_squared_error(Y_train, ypredtrain)
# MSEtest = mean_squared_error(Y_test, ypredtest)
# print("la valeur de r2 d'entrainement pour svr")
# r2_train=1 - (np.sum((Y_train - ypredtrain)**2) / np.sum((Y_train - Y_train.mean())**2))
# print(r2_train)
# print("la valeur de r2 de test  pour svr ")
# r2_test=1 - (np.sum((Y_test - ypredtest)**2) / np.sum((Y_test - Y_test.mean())**2))
# print(r2_test)
# print("la valeur de rmse d'entrainement pour svr")
# RMSEtrain = np.sqrt(MSEtrain)
# print(RMSEtrain)
# print("la valeur de rmse de test pour svr")
# RMSEtest = np.sqrt(MSEtest)
# print(RMSEtest)
# MAEtrain =mean_absolute_error(Y_train,ypredtrain)
# MAEtest = mean_absolute_error(Y_test, ypredtest)
# print('mae svr train',MAEtrain)
# print('mae svr test',MAEtest)
# -------------------cross validation---------------
# -----------------------------lasso----------------
lasso= Lasso()
lasso.fit(X_train,Y_train)

# -----------------------ridge------------------------------
ridgeReg = Ridge()
ridgeReg.fit(X_train, Y_train)

# ------------------------------random forest-----------------
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

# ----------------------------elastic net-----------------
elastic=ElasticNet()
elastic.fit(X_train, Y_train)
# --------------------------svr linéaire----------------
svr_lin=SVR(kernel='linear')
svr_lin.fit(X_train, Y_train)
# --------------------------svr rbf----------------
svr_rbf=SVR(kernel='rbf')
svr_rbf.fit(X_train, Y_train)

modeles= [('lasso', lasso),('ridge',ridgeReg ),('rf',rf),('elastic',elastic),
          ('svr_lin',svr_lin),('svr_rbf',svr_rbf)]
 # ----------la moyenne de r2--------------
resultats= []
for name, model in modeles:
          scores = cross_validate(model, X_train, Y_train, scoring='r2', cv=10, return_train_score=True)
          resultats.append(scores)
r2_lasso=np.mean(resultats[0]['test_score'])
print('moyenne de r2 de lasso',r2_lasso)
r2_ridge=np.mean(resultats[1]['test_score'])
print('moyenne de r2 de ridge',r2_ridge)
r2_rf=np.mean(resultats[2]['test_score'])
print('moyenne de r2 de rf',r2_rf)
r2_elastic=np.mean(resultats[3]['test_score'])
print('moyenne de r2 de elastic',r2_elastic)
r2_svr_lin=np.mean(resultats[4]['test_score'])
print('moyenne de r2 de svr lin',r2_svr_lin)
r2_svr_rbf=np.mean(resultats[5]['test_score'])
print('moyenne de r2 de svr rbf',r2_svr_rbf)
print()
modelDF = pd.DataFrame({
    'Model'       : ['Lasso','ridge','rf','elastic','svr_rbf'],
    'r2_mean'    : [r2_lasso,r2_ridge,r2_rf,r2_elastic,r2_svr_rbf]
    })
sns.factorplot(x= 'Model', y= 'r2_mean', data= modelDF, kind='bar', legend='True')
  # ----------la moyenne de rmse--------------
resultatss= []
for name, model in modeles:
          scores = cross_validate(model, X_train, Y_train, 
          scoring='neg_mean_squared_error', cv=10, return_train_score=True)
          resultatss.append(scores)
mse_lasso=np.mean(resultatss[0]['test_score'])
rmse_lasso=np.sqrt(-mse_lasso)
print('moyenne de rmse de lasso',rmse_lasso)
mse_ridge=np.mean(resultatss[1]['test_score'])
rmse_ridge=np.sqrt(-mse_ridge)
print('moyenne de rmse de ridge',rmse_ridge)
mse_rf=np.mean(resultatss[2]['test_score'])
rmse_rf=np.sqrt(-mse_rf)
print('moyenne de rmse de rf',rmse_rf)
mse_elastic=np.mean(resultatss[3]['test_score'])
rmse_elastic=np.sqrt(-mse_elastic)
print('moyenne de rmse de elastic',rmse_elastic)
mse_svr_lin=np.mean(resultatss[4]['test_score'])
rmse_svr_lin=np.sqrt(-mse_svr_lin)
print('moyenne de rmse de svr lin',rmse_svr_lin)
mse_svr_rbf=np.mean(resultatss[5]['test_score'])
rmse_svr_rbf=np.sqrt(-mse_svr_rbf)
print('moyenne de rmse de svr rbf',rmse_svr_rbf)
print()

modelDF = pd.DataFrame({
    'Model'       : ['Lasso','ridge','rf','elastic','svr_rbf'],
    'rmse_mean'    : [rmse_lasso,rmse_ridge,rmse_rf,rmse_elastic,rmse_svr_rbf]
    })
sns.factorplot(x= 'Model', y= 'rmse_mean', data= modelDF, kind='bar', legend='True')
  # ----------la moyenne de mae--------------
resultatsss= []
for name, model in modeles:
          scores = cross_validate(model, X_train, Y_train, 
          scoring='neg_mean_absolute_error', cv=10, return_train_score=True)
          resultatsss.append(scores)
mae_lasso=np.mean(resultatsss[0]['test_score'])
print('moyenne de mae de lasso',-mae_lasso)
mae_ridge=np.mean(resultatsss[1]['test_score'])
print('moyenne de mae de ridge',-mae_ridge)
mae_rf=np.mean(resultatsss[2]['test_score'])
print('moyenne de mae de rf',-mae_rf)
mae_elastic=np.mean(resultatsss[3]['test_score'])
print('moyenne de mae de elastic',-mae_elastic)
mae_svr_lin=np.mean(resultatsss[4]['test_score'])
print('moyenne de mae de svr linéaire',-mae_svr_lin)
mae_svr_rbf=np.mean(resultatsss[5]['test_score'])
print('moyenne de mae de svr rbf',-mae_svr_rbf)
print()

modelDF = pd.DataFrame({
    'Model'       : ['Lasso','ridge','rf','elastic','svr_rbf'],
    'mae_mean'    : [-mae_lasso,-mae_ridge,-mae_rf,-mae_elastic,-mae_svr_rbf]
    })
sns.factorplot(x= 'Model', y= 'mae_mean', data= modelDF, kind='bar', legend='True')