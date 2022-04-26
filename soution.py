##Import Libraries and Data##
import pandas as pd
import matplotlib.pyplot as plt


dev=pd.read_csv("dev.tsv",sep='\t')
eval=pd.read_csv("eval.tsv",sep='\t')
fig,ax=plt.subplots(1,2)
plt.tight_layout()
ax[0].hist(dev['quality'].values,bins='auto')
#I can see that there are few values
def RemoveOutliars(dev,a,b):
    dev=dev.loc[(dev['quality'] >= a) & (dev['quality'] <= b)]
    ax[1].hist(dev['quality'].values,bins='auto')
    #plt.show()
    return dev
dev=RemoveOutliars(dev,10,90)
#Remove "description" feature and vectorize##
dev.pop("description")
eval.pop("description")
##Sample##
def sample(df,frac):
    df=df.sample(frac=frac,random_state=42)
    return df
#dev=sample(dev,0.1)

##Count number of different values for each feature##
keys=list(dev.keys())
print(keys)

# for key in keys:
#     if key!='quality':
#         print(str.upper(key))
#         print("#distinct values: "+str(dev[key].nunique()))
#         print("#Non_null values: "+str(dev[key].count()))
#         print("#Null Values: " + str(dev[key].size - dev[key].count()))

##7k/12k of region_2 are Nan ==> Remove the entire feature
#dev.pop("region_2")
#eval.pop("region_2")
#keys.remove("region_2")
keys.remove("quality")


#FILL EACH NaN with the most frequent value of the column
#dev = dev.fillna(dev.mode().iloc[0])



##ECONDING
#Fit and trasform in the whole dataset (dev+eval) to encode in the rigth way
tot=pd.concat([dev,eval])
import category_encoders as ce
for key in keys:
    encoder = ce.BinaryEncoder(cols=[key], return_df=True)
    #encoder = ce.BaseNEncoder(cols=[key], return_df=True, base=4) #SAME RESULT OF BINARY
    tot= encoder.fit_transform(tot)
print(tot)

len_dev=len(dev.iloc[:,0])
dev_fin=tot.iloc[:len_dev]
eval_fin=tot.iloc[len_dev:]

##CREATE THE FINAL DATASET AND SPLIT
y_dev=dev_fin.pop("quality")
X_dev=dev_fin
eval_fin.pop("quality")
X_eval=eval_fin

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)

##IMPORT MODELS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

#PARAM GRID FUNCTION
def choose_model(name):
    if name=='Ridge':
        param_grid={
                    'alpha':[1,0.2,0.5],
                    'normalize':[True,False],
                    'fit_intercept':[True,False],
                    'tol':[0.2,0.4,0.001],
                    'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                    'random_state':[42,None]
                    }
        reg=Ridge()
    elif name=='LinearRegression':
        param_grid={
                    'fit_intercept':[True,False],
                    'normalize':[True,False],
                    'copy_X':[True,False],
                    'n_jobs':[None,-1,5,10,50]
                    }
        reg=LinearRegression()
    elif name=='SVR':
        param_grid={
                    'kernel':['linear','poly','rbf','sigmoid','precomputed'],
                    'degree':[2,3,5],
                    'gamma':['scale','auto',1.0],
                    'tol':[0.4,0.6],
                    #'epsilon':[0.1,0.5,1.0],
                    #'shrinking':[True,False]
                    }
        reg=SVR()
    elif name=='MLPRegressor':
        param_grid={
                    'activation':['logistic'],
                    #'solver':['lbfgs','sgd','adam'],
                    #'alpha':[0.001,0.1,0.5,1],
                    #'learning_rate':['constant','invscaling','adaptive'],
                    'max_iter':[1000]
                    #'tol':[0.4,0.8],
                    #'epsilon':[0.00001,0.3,1.0]
                    }
        reg=MLPRegressor()
    elif name=='RandomForestRegressor':
        param_grid={
                    'n_estimators':[100,500,900],
                    'max_depth':[None,2,10,30,50],
                    'criterion':['mse'],
                    'random_state':[42],
                    'n_jobs':-1
                    }
        reg=RandomForestRegressor()
    elif name=='Ridge_N':
        param_grid={'alpha':[0.1,0.2,0.5]}
        reg=Ridge()
    elif name=='LinearRegression_N':
        param_grid={}
        reg=LinearRegression()
    elif name=='SVR_N':
        param_grid={}
        reg=SVR()
    elif name=='MLPRegressor_N':
        param_grid={}
        reg=MLPRegressor()
    elif name=='RandomForestRegressor_N':
        param_grid={}
        reg=RandomForestRegressor(#random_state=42,
                                  #BEST CONFIGURATION SO FAR
                                  #max_depth=30,n_estimators=950,criterion='mae'
                                                   )
    else:
        print('errore')
    return param_grid,reg

# CHOOSE 1 MODEL AND TUNE IS HYPERPARAM

# NAIVE ==> WITHOUT PARAMETER
#param_grid,reg=choose_model('LinearRegression_N') #0.16
# param_grid,reg=choose_model('Ridge_N') #==> 0.19......... alpha=0.5
#param_grid,reg=choose_model('SVR_N') #==> 0.347 ok
#param_grid,reg=choose_model('MLPRegressor_N') #==>ok 0.41 without sampling and 0.22 with 0.1 frac
#param_grid,reg=choose_model('RandomForestRegressor_N') #==>0.61 ok ==>0.79 without sampling !!!!
# RANDOM FOREST ==> 0.47 con sample al 0.1

# WITH GRID PARAMETER
# param_grid,reg=choose_model('LinearRegression')
# param_grid,reg=choose_model('Ridge')
#param_grid,reg=choose_model('SVR')
#param_grid,reg=choose_model('MLPRegressor')
param_grid,reg=choose_model('RandomForestRegressor')

from sklearn.model_selection import GridSearchCV
gridsearch = GridSearchCV(reg,param_grid,scoring='r2',cv=3)

res=gridsearch.fit(X_train,y_train)

print(gridsearch.best_params_)
print(gridsearch.best_estimator_)

#CHOOSE THE BEST MODEL
best_model=gridsearch.best_estimator_
best_model.fit(X_train,y_train)

#CALCULATE R2 SCORE
y_test_pred=best_model.predict(X_test)
from sklearn.metrics import r2_score
print("BEST R2: ",r2_score(y_test,y_test_pred))

def final_prediction():
    best_model.fit(X_dev,y_dev)
    y_pred =(best_model.predict(X_eval)) #NO ROUND
    final=pd.DataFrame(y_pred)
    final.rename(columns={0:"Predicted" }, inplace = True)
    #SAVE FINAL CSV
    final.to_csv('./submission.csv',index=True,index_label='Id')
    return
final_prediction()


