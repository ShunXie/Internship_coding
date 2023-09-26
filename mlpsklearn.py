#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, metrics, Model, Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, GaussianNoise, Activation
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, auc
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[2]:


get_ipython().system(u'source activate python37 && pip3 install fastparquet -i https://pypi.douban.com/simple')


# ## 1 考虑用户特征

# In[3]:


from nbsdk import get_table, get_pandas
table_mkt = get_table('5d91a04b-5d71-4fed-87cc-72c8696e053c/sx_churn_feature_mkt.table')
df_mkt = table_mkt.to_pandas()
df_mkt['churn_100'] = df_mkt['churn_100'].astype(np.int64)

X_tmp_100_mkt,y_100_mkt = df_mkt.iloc[:,:-4], df_mkt.iloc[:,-4]
y_100_mkt =  np.array(y_100_mkt)
X_100_mkt = pd.get_dummies(X_tmp_100_mkt)
X_train_100_mkt, X_test_100_mkt, y_train_100_mkt, y_test_100_mkt = train_test_split(X_100_mkt,y_100_mkt,test_size = 0.25, random_state = 40)
scaler = preprocessing.StandardScaler().fit(X_train_100_mkt)
X_train_100_mkt = scaler.transform(X_train_100_mkt)
X_test_100_mkt = scaler.transform(X_test_100_mkt)
y_train_100_mkt = np.array([[y for y in y_train_100_mkt]]).T
y_test_100_mkt = np.array([[y for y in y_test_100_mkt]]).T


# In[3]:


# 定义MLP模型
def MLP(input_dim, output_dim, lr = 0.1):
    input = Input(shape=input_dim)
    # 批归一化
    x = BatchNormalization()(input)
    hidden_units = [60, 30]
    for id, hidden_unit in enumerate(hidden_units):
        x = Dense(hidden_unit)(x)
        x = BatchNormalization()(x)
        x = Lambda(tf.keras.activations.tanh)(x)
        x = Dropout(0.2)(x)

    x = Dense(output_dim, activation='sigmoid')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.BinaryCrossentropy(label_smoothing=0.05),
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model




# In[4]:


def mlp_cv(X_train, y_train):
    lr = [0.1,0.05,0.01]
    result = []
    num_folds = 5
    #define cross validation
    kf = KFold(n_splits = num_folds, shuffle = True)
    cv_results = []

    for lr_val in lr:

        fold_scores = []
        print("start training")
        for train_index, val_index in kf.split(X_train):
            # Split data into train and validation sets for each fold
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

            model = MLP(input_dim=X_train.shape[1], output_dim=1, lr = lr_val)

            # 训练模型
            # 早停keras.callbacks.EarlyStopping
            es = keras.callbacks.EarlyStopping(monitor='val_auc', patience=10)
            history = model.fit(X_train, y_train, batch_size=128, epochs=100, validation_split=0.2,
                                callbacks=[es], verbose=2)
            # Evaluate the model on the validation set
            y_val_pred = model.predict(X_val)
            fold_score = roc_auc_score(y_val, y_val_pred)
            fold_scores.append(fold_score)


    # Calculate the average score across all folds for the current learning rate
    avg_score = np.mean(fold_scores)
    cv_results.append((lr_val, avg_score))
    print(f'learning rate = {lr_val}的测试集的auc为{avg_score}')

    return cv_results

def best_parameter(cv_results):
    metric_val = [cv_results[i][-1] for i in range(len(cv_results))]
    largest_ind = metric_val.index(max(metric_val))
    return cv_results[largest_ind]




# In[30]:


cv_results = mlp_cv(X_train_100_mkt, y_train_100_mkt)


# In[7]:


model = MLP(input_dim=X_train_100_mkt.shape[1], output_dim=1,lr = best_parameter(cv_results)[0])
es = keras.callbacks.EarlyStopping(monitor='val_auc', patience=20)
history = model.fit(X_train_100_mkt, y_train_100_mkt, batch_size=128, epochs=100, validation_split=0.2,
                    callbacks=[es], verbose=2)


# 迭代打印训练情况
train_auc = history.history['auc']
val_auc = history.history['val_auc']
iter_epoch = len(train_auc)
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.plot(np.arange(iter_epoch), train_auc, label='train_auc')
ax.plot(np.arange(iter_epoch), val_auc, label='val_auc')
ax.set_title('train iter auc')
ax.set_ylabel('AUC')
ax.set_xlabel('epoch')
ax.legend(loc='upper right')
plt.show()

# 模型预测
y_pred = model.predict(X_test_100_mkt)
auc = roc_auc_score(y_test_100_mkt, y_pred)
print('测试集的auc为', auc)


# In[39]:


train_auc = history.history['loss']
val_auc = history.history['val_loss']
iter_epoch = len(train_auc)
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.plot(np.arange(iter_epoch), train_auc, label='train_loss')
ax.plot(np.arange(iter_epoch), val_auc, label='val_loss')
ax.set_title('train iter loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(loc='upper right')
plt.show()


# ## 2. 加入coupon以及mkt值以后

# 
# from nbsdk import get_table, get_pandas
# table_new = get_table('bc8a0537-b526-4f10-acd1-6fbd85cbe2b7/sx_churn_feature_coupon_new.table')
# df_new = table_new.to_pandas()
# df_new['churn_100'] = df_new['churn_100'].astype(np.int64)
# df_new = df_new.fillna(-1)
# 
# 
# 
# from nbsdk import get_table, get_pandas
# table_new = get_table('bc21aa44-f11e-41fd-83fa-a463a49013e8/sx_distinct_feature_with_diff.table')
# df_new = table_new.to_pandas()

# In[5]:



from nbsdk import get_table, get_pandas
table_new = get_table('e92229cd-c6a7-4389-8d31-31d3d9970478/churn_feat_diff_final.table')
df_new = table_new.to_pandas()
df_new['churn_100'] = df_new['churn_100'].astype(np.int64)
df_new = df_new.fillna(-1)
df_new.loc[:,"ta_diff"]=df_new.loc[:,"ta_diff"].astype("float")
df_new.loc[:,"ta_diff"]=df_new.loc[:,"tc_diff"].astype("float")


X_tmp_100_new,y_100_new = df_new.iloc[:,1:-4], df_new.iloc[:,-4]
y_100_new =  np.array(y_100_new)
X_100_new = pd.get_dummies(X_tmp_100_new)
X_train_100_new, X_test_100_new, y_train_100_new, y_test_100_new = train_test_split(X_100_new,y_100_new,test_size = 0.25, random_state = 40)
scaler = preprocessing.StandardScaler().fit(X_train_100_new)
X_train_100_new = scaler.transform(X_train_100_new)
X_test_100_new = scaler.transform(X_test_100_new)
y_train_100_new = np.array([[y for y in y_train_100_new]]).T
y_test_100_new = np.array([[y for y in y_test_100_new]]).T


# In[6]:


cv_results = mlp_cv(X_train_100_new, y_train_100_new)


# In[8]:


best_parameter(cv_results)


# In[9]:


model = MLP(input_dim=X_test_100_new.shape[1], output_dim=1,lr = best_parameter(cv_results)[0])
es = keras.callbacks.EarlyStopping(monitor='val_auc', patience=20)
history = model.fit(X_train_100_new, y_train_100_new, batch_size=128, epochs=100, validation_split=0.2,
                    callbacks=[es], verbose=2)


# 迭代打印训练情况
train_auc = history.history['auc']
val_auc = history.history['val_auc']
iter_epoch = len(train_auc)
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.plot(np.arange(iter_epoch), train_auc, label='train_auc')
ax.plot(np.arange(iter_epoch), val_auc, label='val_auc')
ax.set_title('train iter auc')
ax.set_ylabel('AUC')
ax.set_xlabel('epoch')
ax.legend(loc='upper right')
plt.show()

# 模型预测
y_pred = model.predict(X_test_100_new)
auc = roc_auc_score(y_test_100_new, y_pred)
print('测试集的auc为', auc)


# In[10]:


train_auc = history.history['loss']
val_auc = history.history['val_loss']
iter_epoch = len(train_auc)
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111)
ax.plot(np.arange(iter_epoch), train_auc, label='train_loss')
ax.plot(np.arange(iter_epoch), val_auc, label='val_loss')
ax.set_title('train iter loss')
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(loc='upper right')
plt.show()