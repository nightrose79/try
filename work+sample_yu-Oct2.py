
# coding: utf-8

import os
os.chdir('/Users/yuy/Documents/life/resume/Amazon/')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
import random
from scipy import stats
from sklearn import svm
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve, auc#,precision_recall_curve,average_precision_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,TimeSeriesSplit


##############################
#parameters
#ratio_of_failed_to_not=0.2
#pred_freq='daily' #'weekly' 
random.seed(5678)
#sample_split='time_series'
s_window=7
l_window=30
weight=1

device_failure=pd.read_csv('device_failure.csv')

#######################################
# # I. Data Exploration
#######################################
device_failure.date=pd.to_datetime(device_failure.date)
#the time window for the data is 2015-01-01 to 2015-11-02

#failure frequency by device

device_failure.groupby('device').failure.sum().value_counts()#each device has at most one failure.
#those with and without failure number 1:10

#list the devices with failures
failed=device_failure[device_failure['failure']==1].device
failed.reset_index(drop=True,inplace=True)
device_failure.loc[device_failure['device']==failed[3],['date','failure']] 


df=device_failure.reset_index()
devices=df.device.unique()
#devices[sample(0.2)]
full=set(devices)
n=len(full)
print('full set has %d items'%(n))


#######################################
# # II.Feature engineering
#######################################

#create useful features from the 9 attributes

#I calculate rolling mean (and rolling std for attribute 1)while keeping the window of mean calculation as a variable 
#that we could change.


for i in range(1,10):
    device_failure['attribute'+str(i)+'_s_rolling_mean']=pd.rolling_mean(device_failure['attribute'+str(i)], window=s_window, min_periods=5)#.mean()
    device_failure['attribute'+str(i)+'_l_rolling_mean']=pd.rolling_mean(device_failure['attribute'+str(i)], window=l_window, min_periods=30)#.mean()

    if (i in (1,6)):
        device_failure['attribute'+str(i)+'_s_rolling_std']=pd.rolling_std(device_failure['attribute'+str(i)], window=s_window, min_periods=5)#.std()
        device_failure['attribute'+str(i)+'_l_rolling_std']=pd.rolling_std(device_failure['attribute'+str(i)], window=l_window, min_periods=30)#.std()

device_failure.tail() #check variable creation results

#changes from initial value
temp=device_failure.copy()
temp['initial_date']=temp.groupby('device')['date'].transform('min')
first_record=temp[temp['date']==temp['initial_date']]

for i in range(1,10):
    first_record=first_record.rename(columns={'attribute'+str(i):'initial_attribute'+str(i)})

first_record.drop(['failure','date','initial_date',
                   'attribute1_s_rolling_mean','attribute2_s_rolling_mean',
                   'attribute3_s_rolling_mean','attribute4_s_rolling_mean','attribute5_s_rolling_mean',
                   'attribute6_s_rolling_mean','attribute7_s_rolling_mean','attribute8_s_rolling_mean',
                   'attribute9_s_rolling_mean',
                   'attribute1_s_rolling_std','attribute6_s_rolling_std',
                   'attribute1_l_rolling_mean','attribute2_l_rolling_mean',
                   'attribute3_l_rolling_mean','attribute4_l_rolling_mean','attribute5_l_rolling_mean',
                   'attribute6_l_rolling_mean','attribute7_l_rolling_mean','attribute8_l_rolling_mean',
                   'attribute9_l_rolling_mean',
                   'attribute1_l_rolling_std','attribute6_l_rolling_std'],axis=1,inplace=True)


diff=pd.merge(device_failure,first_record,how='left',on='device')    
for i in range(1,10):
    diff['attribute'+str(i)+'_diff']=diff['attribute'+str(i)]-diff['initial_attribute'+str(i)]

diff.drop(['initial_attribute1','initial_attribute2',
                   'initial_attribute3','initial_attribute4','initial_attribute5',
                   'initial_attribute6','initial_attribute7','initial_attribute8',
                   'initial_attribute9'],axis=1,inplace=True)



#single period change of attributes 

diff.set_index(['device','date'],inplace=True)
diff.sort_index(inplace=True)
shift=diff.shift(1,axis=0) #move the previous period x to the next period, since we can only use 
#last period values for prediction

for i in range(1,10):
    shift=shift.rename(columns={'attribute'+str(i):'lag_attribute'+str(i)})
    
shift.drop(['failure','attribute1_s_rolling_mean','attribute2_s_rolling_mean',
                   'attribute3_s_rolling_mean','attribute4_s_rolling_mean','attribute5_s_rolling_mean',
                   'attribute6_s_rolling_mean','attribute7_s_rolling_mean','attribute8_s_rolling_mean',
                   'attribute9_s_rolling_mean',
                   'attribute1_s_rolling_std','attribute6_s_rolling_std',
                   'attribute1_l_rolling_mean','attribute2_l_rolling_mean',
                   'attribute3_l_rolling_mean','attribute4_l_rolling_mean','attribute5_l_rolling_mean',
                   'attribute6_l_rolling_mean','attribute7_l_rolling_mean','attribute8_l_rolling_mean',
                   'attribute9_l_rolling_mean',
                   'attribute1_l_rolling_std','attribute6_l_rolling_std',
                   
                   'attribute1_diff','attribute2_diff','attribute3_diff','attribute4_diff',
                   'attribute5_diff','attribute6_diff','attribute7_diff','attribute8_diff',
                   'attribute9_diff'],axis=1,inplace=True)

diff2=pd.merge(diff,shift,how='left',right_index=True,left_index=True)
for i in range(1,10):
    diff2['attribute'+str(i)+'_change']=diff2['attribute'+str(i)]-diff2['lag_attribute'+str(i)]

#first shift next period y up for prediction
results1=diff2.shift(-1,axis=0)
results1=results1['failure']
results_1=results1.to_frame(name='failure_next_period')

modeling_set=pd.merge(diff2,results_1,how='left',right_index=True,left_index=True)
modeling_set.drop(['failure'],axis=1,inplace=True)

final_modeling_set=modeling_set.copy()
final_modeling_set.rename(columns={'failure_next_period':'y'},inplace=True)
final_modeling_set.reset_index(inplace=True)

#######################################
# # III.Modeling Fitting
#######################################

#creating training, validation and testing sample

#first find out the number of failures per month
failure_by_month=device_failure.set_index(['date']).resample('M',label='right',convention='end',level='date').failure.sum()
failure_by_month.plot()
#we use first 6 months for training and validation, and the last 5 months for testing
train_val_sample=final_modeling_set[final_modeling_set.date<'2015-07-01']
train_indices=train_val_sample[train_val_sample.date<'2015-02-01'].index


testing_sample=final_modeling_set[final_modeling_set.date>='2015-07-01']
    
removal_list=['lag_attribute1','lag_attribute2','lag_attribute3','lag_attribute4',
              'lag_attribute5','lag_attribute6','lag_attribute7','lag_attribute8',
              'lag_attribute9']

train_val_sample.dropna(inplace=True)
train_val_sample.reset_index(drop=True,inplace=True)
testing_sample.dropna(inplace=True)
X_train=train_val_sample.drop(['y','date','device']+removal_list,1)
y_train=train_val_sample['y'].astype(int)
X_test=testing_sample.drop(['y','date','device']+removal_list,1)
y_test=testing_sample['y'].astype(int)

y_train.value_counts()

#I create 3 training samples and 3 validation samples. 


tscv=TimeSeriesSplit(n_splits=3)
print(tscv)

for train,test in tscv.split(X_train):
    print('%s %s' %(train,test))
    
################################################
#fit the model

#Cross validation and hyper-parameter search


print('running cross validation')

########################################
#XGBoost

clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')
param_dist_xgb = {'n_estimators': stats.randint(150, 500),
              'learning_rate': stats.uniform(0.01, 0.07),
              'subsample': stats.uniform(0.3, 0.7),
              'max_depth': [3, 6,  9],
              'colsample_bytree': stats.uniform(0.5, 0.45),
              'min_child_weight': [1, 2, 3],
              'scale_pos_weight':[weight]
              
             }
clf = RandomizedSearchCV(clf_xgb, param_distributions = param_dist_xgb, 
                         n_iter = 25, scoring = 'roc_auc', error_score = 0, 
                         cv=tscv, verbose = 3, n_jobs = -1,
                         refit=True)
clf.fit(X_train,y_train,early_stopping_rounds=10)

#examine
clf.cv_results_

print(clf.best_estimator_)

print(clf.best_score_)

best_param=clf.best_params_

#clf.predict(X_test)
preds_prob=clf.predict_proba(X_test)
#find the optimum hyper-parameters, apply them to the overall model training

preds_train_prob = clf.predict_proba(X_train)
preds_train = preds_train_prob[:,1]
print('Roc-auc for train sample is %.2f' %(roc_auc_score(y_train,preds_train)))

preds=preds_prob[:,1]
print('Roc-auc for test sample is %.2f' %(roc_auc_score(y_test,preds)))
########################################
#SVC

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf_svc = GridSearchCV(estimator=svc, param_grid=param_grid,scoring='roc_auc',
                       n_jobs=-1,cv=tscv,verbose=3,refit=True)
clf_svc.fit(X_train,y_train)

clf_svc.cv_results_

print(clf_svc.best_estimator_)

print(clf_svc.best_score_)

best_param=clf_svc.best_params_

preds_train_svc_prob = clf_svc.predict_proba(X_train)
preds_train_svc = preds_train_svc_prob[:,1]
print('Roc-auc for train sample is %.2f' %(roc_auc_score(y_train,preds_train_svc)))

preds_test_svc_prob = clf_svc.predict_proba(X_test)
preds_svc=preds_test_svc_prob[:,1]
print('Roc-auc for test sample is %.2f' %(roc_auc_score(y_test,preds_svc)))
#######################################
# # IV. Performance Check
#######################################

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=1,
             label='ROC (AUC = %0.2f)' % (roc_auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve--AUC for testing sample is %.2f' % (roc_auc_score(y_test,preds)))
plt.show()

roc_df=pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})
roc_df['error_rate_for_positives']=roc_df.fpr
roc_df['error_rate_for_negatives']=0
roc_df['tp']=0
roc_df['tn']=0
roc_df['fp']=0
roc_df['fn']=0
roc_df['tp_device']=0
roc_df['tn_device']=0
roc_df['fp_device']=0
roc_df['fp_device']=0
roc_df['device_error_rate_for_positives']=0
roc_df['device_error_rate_for_negatives']=0

temp=testing_sample.copy()


for n in range(roc_df.shape[0]):
    i=roc_df.loc[n,'thresholds']
    preds_01 = preds >=i#/10.0
    preds_01=preds_01 *1
    cnf_matrix = confusion_matrix(y_test, preds_01)#.ravel()
    tn,fp,fn,tp = cnf_matrix.ravel()
    roc_df.loc[n,'tp']=tp
    roc_df.loc[n,'fp']=fp
    roc_df.loc[n,'tn']=tn
    roc_df.loc[n,'fn']=fn
    roc_df.loc[n,'error_rate_for_positives'] = fp/(fp+tn)
    roc_df.loc[n,'error_rate_for_negatives'] = fn/(fn+tp)
    
    temp.loc[:,'pred']=preds_01
    accuracy=temp.groupby('device')[['y','pred']].sum()
    accuracy.loc[:,'predicted']=accuracy.pred>0
    a=pd.crosstab(accuracy.y,accuracy.predicted)
    if a.shape[1]==1:
        a.loc[:,'True']=0
    tn_device,fp_device,fn_device,tp_device=a.values.ravel()
    roc_df.loc[n,'tp_device']=tp_device
    roc_df.loc[n,'fp_device']=fp_device
    roc_df.loc[n,'tn_device']=tn_device
    roc_df.loc[n,'fn_device']=fn_device
    roc_df['device_error_rate_for_positives']=fp_device/(tn_device+fp_device)
    roc_df['device_error_rate_for_negatives']=fn_device/(fn_device+tp_device)
    

#plot the error rate tradeoff between the two classes. 
roc_df.sort_values('error_rate_for_positives',inplace=True)
plt.plot(roc_df.error_rate_for_positives, roc_df.error_rate_for_negatives, lw=1, alpha=1,
             label='Error Rate Trade-offs for two classes')
plt.xlabel('False positive rate')
plt.ylabel('False negative rate')
plt.title('Error Rate Trade-offs for two classes' )
plt.show()

roc_df.diff=np.abs(roc_df.error_rate_for_negatives - roc_df.error_rate_for_positives)
minimum_error_rate=roc_df.loc[roc_df.diff==min(roc_df.diff),'error_rate_for_negatives']

plt.plot(roc_df.error_rate_for_positives,label='error rate for positives')
plt.plot(roc_df.error_rate_for_negatives,label='error rate for negatives')
plt.plot(roc_df.diff,label='difference of the two')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.title('The minimum error rate for both classes are %.2f' % (minimum_error_rate))
plt.show()
 
#suppose that the cost of checking 100 good devices is equal to the cost of missing one failing device
roc_df.trade_off=roc_df.fp-100*roc_df.tp
fp_cross=0
for i in range(len(roc_df.trade_off)-3):
    
    current=roc_df.trade_off[i]
    next_one = roc_df.trade_off[i+1]
    next_two = roc_df.trade_off[i+2]
    next_three = roc_df.trade_off[i+3]
    if (current<0 and next_one >0 and next_two>0 and next_three >0): 
        fp_cross=roc_df.fp[i]
        
#fp_cross=roc_df.loc[np.abs(roc_df.trade_off)==np.min(np.abs(roc_df.trade_off)),'fp']#.value

plt.plot(roc_df.fp, roc_df.tp*100, lw=1, alpha=1)
plt.plot([0,2500],[0,2500],ls="--", c="red")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate*100')
plt.axvline(x=fp_cross,color='k',linewidth=0.5)
plt.text(fp_cross,0,'%d'%(fp_cross),rotation=0)
plt.title('Trade-off between True Positive and False Positive')
plt.show()

#the optimal threshold for failure prediction is to refer the top 1035 for checking. It will capture about 10 failures and causes ~ 1000 wrong checks. 

plt.plot(roc_df.fp, roc_df.tp*100, lw=1, alpha=1)
plt.plot([0,2500],[0,2500],ls="--", c="red")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate*100')
plt.axvline(x=fp_cross,color='k',linewidth=0.5)
plt.text(fp_cross,0,'%d'%(fp_cross),rotation=0)
plt.title('Trade-off between True Positive and False Positive (Zoomed in)')
plt.xlim([0,2500])
plt.ylim([0,2500])
plt.show()

def lift(sortby, pred, actual, weight=None, savePath="e:\\mydata\\",n=10, plot=True, std=None, title="Gini", show=True):
    """
    sort, pred, actual, weight can be a 1-d numpy array or a single column of a pandas dataframe
    sortby = sort by variable
    pred = predicted values
    actual = actual values
    weight = weights (leave blank if NA)
    n = number of buckets
    plot = True if plot the resulting lift chart
    std = if not None, shows the error bar for each bucket
    """
    if weight is None:
        weight = np.ones_like(pred)

    def weighted_std(values, weight):
        """
        weighted std of an array
        this is slightly biased but shouldn't matter for big n
        """
        m = np.average(values, weights=weight)
        return np.sqrt(np.average((values-m)**2, weights=weight))

    r = np.vstack((sortby, pred, actual, weight)).T
    r = r[sortby.argsort()].T
    cumm_w = np.cumsum(r[3])
    cumm_y = np.cumsum(r[2]*r[3])
    total_w = np.sum(weight)
    gini = 1-2*(np.sum(cumm_y*r[3])/(np.sum(r[2]*r[3])*total_w))
    idx = np.clip(np.round(cumm_w*n/total_w + 0.5), 1, n) - 1
    lift_chart = np.zeros((n,7))
    for i in range(n):
        lift_chart[i][0] = np.sum(r[3][idx==i]) #num observations in each bucket
        lift_chart[i][1] = np.sum(r[1][idx==i]*r[3][idx==i])/lift_chart[i][0] #mean prediction
        lift_chart[i][2] = np.sum(r[2][idx==i]*r[3][idx==i])/lift_chart[i][0] #mean actual
        lift_chart[i][3] = weighted_std(r[1][idx==i],r[3][idx==i]) #weighted std
        lift_chart[i][4] = np.average(r[0][idx==i],weights=r[3][idx==i])#mean sortby variable
        lift_chart[i][5] = np.min(r[0][idx==i]) #min sortby variable
        lift_chart[i][6] = np.max(r[0][idx==i])#max sortby variable
    if plot==True:
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        x = range(1,n+1)
        ax.plot(x, lift_chart[:,1], "b", label="Predicted")
        ax.plot(x,  lift_chart[:,2], "r", label="Actual")
        ax.grid(True)
        ax.set_xlabel("Buckets (Equal Exposure)")
        ax.set_ylabel("Failure Probability by Prediction Buckets")
        ax.axhline(y=actual.mean(),c='black',linestyle='--')
        ax.text(0,actual.mean()*1.5,'chance prediction prob is %.5f'%(actual.mean()),rotation=0)
        ax.legend(loc=2)
        if std is not None:
            ax.fill_between(x, lift_chart[:,1]+std*lift_chart[:,3],lift_chart[:,1]-std*lift_chart[:,3],
                            color="b", alpha=0.1)
        #ax.set_title(title + "\n Gini: " + format(gini, ".4f"))
        ax.set_title('Lift Curve')
        fig.savefig(savePath + title + " Lift Chart.png")
        if(show==True):
            plt.show()
        plt.close(fig)
    return gini, lift_chart

lift(preds,preds,y_test,n=25)

#plot confusion matrix
th_cross=roc_df.loc[roc_df.fp==fp_cross,'thresholds'].values[0]
preds_01 = preds >=th_cross
preds_01=preds_01 *1
cnf_matrix = confusion_matrix(y_test, preds_01)#.ravel()

ax=sns.heatmap(cnf_matrix,annot=True,fmt='d',cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
#by adopting the 100 false positive to one true positive cost calculation, by referring the top 1000 or so model predictions, we 
#capture more than half (12) true failure incidences.

#additional check at device level

analysis=testing_sample.copy()
analysis.loc[:,'pred']=preds_01
accuracy=analysis.groupby('device')[['y','pred']].sum()
accuracy.loc[:,'predicted']=accuracy.pred>0
pd.crosstab(accuracy.y,accuracy.predicted)#.ravel()

#os.chdir('/Users/yuy/Documents/Projects/Loss Reserving/Model/code/')
#from utility import printFeatureImportanceXGBoostFeatureList,create_precision_recall_curve,lift,classification_metrics,mygini

