
# coding: utf-8

# In[7]:


from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
#from rgf.sklearn import RGFClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name=['Date','Address','Species','Block','Street','Trap','AddressNumberAndStreet','Latitude','AddressAccuracy','NumMosquitos','WnvPresent']

def get_set(filename):
    train_set=pd.read_csv(filename, header=0)


    train = datasets.load_iris()
    train['data'] =np.array([#pd.Categorical(train_set['Species']).codes, 
       
        train_set['NumMosquitos'],
        train_set['CULEX ERRATICUS'],
        train_set['CULEX PIPIENS'],
        train_set['CULEX PIPIENS/RESTUANS'],
        train_set['CULEX RESTUANS'],
        train_set['CULEX SALINARIUS'],
        train_set['CULEX TARSALIS'],
        train_set['CULEX TERRITANS'],
        
        train_set['Tmax'], 
        train_set['Tmin'], 
        train_set['Tavg'], 
        train_set['DewPoint'], 
        train_set['Heat'], 
        train_set['Cool'], 
        train_set['StnPressure'], 
        train_set['SeaLevel'], 
        train_set['ResultSpeed'], 
        train_set['ResultDir'], 
        train_set['AvgSpeed'], 
        train_set['Tmax.1'], 
        train_set['Tmin.1'], 
        train_set['Tavg.1'], 
        train_set['DewPoint.1'], 
        train_set['Heat.1'], 
        train_set['Cool.1'], 
        train_set['StnPressure.1'], 
        train_set['SeaLevel.1'], 
        train_set['ResultSpeed.1'], 
        train_set['ResultDir.1'], 
        train_set['AvgSpeed.1'], 
        train_set['Latitude'], 
        train_set['Longitude'], 
        train_set['AddressAccuracy'], 
        
    
                                ]).transpose()

    train['DESCR'] = ''
    train['feature_names'] = ['Date', 'Latitude', 'Latitude', 'AddressAccuracy', 'NumMosquitos']
    train['target'] = np.array(train_set['WnvPresent'])
    train['target_names'] = train['target']
    return train


train = get_set('final_train.csv');

rng = check_random_state(0)
perm = rng.permutation(train.target.size)
test_data = train.data[perm]
test_target = train.target[perm]


rgf = GradientBoostingClassifier(loss='deviance', 
                                 learning_rate=0.0035, 
                                 n_estimators=1000, 
                                 subsample=1.0, 
                                 criterion='friedman_mse', 
                                 min_samples_split=2, 
                                 min_samples_leaf=1, 
                                 min_weight_fraction_leaf=0.0, 
                                 max_depth=7, 
                                 min_impurity_decrease=0.0, 
                                 min_impurity_split=None, 
                                 init=None, random_state=None, 
                                 max_features=None, 
                                 verbose=0, 
                                 max_leaf_nodes=None, 
                                 warm_start=False, 
                                 presort='auto')




fit = rgf.fit(train.data, train.target)

'''rgf_scores = cross_val_score(rgf,
                             train.data,
                             train.target,
                             cv=StratifiedKFold(n_folds))
'''
#rgf_score = sum(rgf_scores)/n_folds
#print('RGF Classfier score: {0:.5f}'.format(rgf_score))
print('done')



mapdata = np.loadtxt("mapdata_copyright_openstreetmap_contributors.txt")
traps = pd.read_csv('train.csv')[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent']]

aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)

plt.figure(figsize=(10,14))
plt.imshow(mapdata, 
           cmap=plt.get_cmap('gray'), 
           extent=lon_lat_box, 
           aspect=aspect)


prediction = fit.predict(train.data)

locations = traps[['Longitude', 'Latitude']].drop_duplicates().values
red = np.array([[traps['Longitude'][i], traps['Latitude'][i]]  
             for i in range(len(traps['WnvPresent'])) if (prediction[i] == 1)])
#red = traps[['Longitude', 'Latitude']].drop_duplicates().values
plt.scatter(locations[:,0], locations[:,1], marker='x')
plt.scatter(red[:,0], red[:,1], marker='o', alpha = 0.05, c = 'red', s = 1000)
plt.savefig('estimation.png')
print('done')


# In[38]:


from sklearn import cross_validation,metrics
from sklearn import svm

train_x,test_x,train_y,test_y = cross_validation.train_test_split(train.data,train.target,test_size=0.2,random_state=27)
rgf.fit(train_x,train_y)
predict_prob_y = rgf.predict_proba(test_x)
test_auc = metrics.roc_auc_score(test_y,predict_prob_y[:,1])
print (test_auc)

