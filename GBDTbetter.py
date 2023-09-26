#!/usr/bin/env python
# coding: utf-8

# # GBDT模型取重要特征值

# In[5]:


#导入包
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import copy as cp
from pyhive import hive
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold
import shap



# In[3]:


get_ipython().system(u'source activate python37 && pip3 install fastparquet -i https://pypi.douban.com/simple')


# ## GBDT 模型定义以及调参

# In[10]:



#define gbdt for CV to choose learning rate and max_depth_trees
def cross_val_gbm( X_train, y_train,num_folds=5):

    #define cross validation
    kf = KFold(n_splits = num_folds, shuffle = True)
    cv_results = []

    #cross validation for learning rate, max_depth tree 
    learning_rates = [0.1, 0.05, 0.01]
    max_depth_trees = [2,5,None]

    # Set the parameters for the LightGBM model
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    for learning_rate in learning_rates:
        for max_depth_tree in max_depth_trees:
            if max_depth_tree is None:
                num_leaves_val = 31
            else:
                num_leaves_val = max(2,2**max_depth_tree-20)

            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth_tree
            params['num_leaves'] = num_leaves_val,
            fold_scores = []
            print("start training")
            for train_index, val_index in kf.split(X_train):
                # Split data into train and validation sets for each fold
                X_tr, X_val = X_train[train_index], X_train[val_index]
                y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

                # Create LightGBM datasets for train and validation sets
                train_set = lgb.Dataset(X_tr, label=y_tr)
                val_set = lgb.Dataset(X_val, label=y_val)

                # Train the LightGBM model
                model = lgb.train(params, train_set, num_boost_round=500, 
                   early_stopping_rounds = 20, valid_sets=[val_set], verbose_eval=False)

                # Evaluate the model on the validation set
                y_val_pred = model.predict(X_val)
                fold_score = roc_auc_score(y_val, y_val_pred)
                fold_scores.append(fold_score)

            # Calculate the average score across all folds for the current learning rate
            avg_score = np.mean(fold_scores)
            cv_results.append((learning_rate, max_depth_tree, avg_score))
            print(f'learning rate: {learning_rate}, max depth: {max_depth_tree} is trained, with auc {avg_score}')

    return cv_results

def best_parameter(cv_results):
    metric_val = [cv_results[i][-1] for i in range(len(cv_results))]
    largest_ind = metric_val.index(max(metric_val))
    return cv_results[largest_ind]


# In[139]:



def lgb_train_function(X_train, y_train, X_test, y_test, learning_rate=0.05, max_depth_tree=None, num_tree = 100,return_pred=False, print_plot = False, threshold = 0.5):
    #初始化lgb数据
    #X_tr, X_te, y_tr, y_te = train_test_split(X_train,y_train,test_size = 0.25, random_state = 111)

    #lgb_train_train = lgb.Dataset(X_tr, y_tr, free_raw_data = False)
    #lgb_train_test = lgb.Dataset(X_te, y_te, free_raw_data = False)
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,free_raw_data=False)

    #define some parameters
    _,num_feature =X_train.shape

    #初始化parameter
    if max_depth_tree is None:
        num_leaves_val = 31
    else:
        num_leaves_val = max(2,2**max_depth_tree-20)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': num_leaves_val,
        'learning_rate': learning_rate,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'max_depth': max_depth_tree
    }

    #显示feature的名字
    feature_name = ['feature_' + str(col) for col in range(num_feature)]

    #训练
    print('开始训练...')
    eval_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_train,lgb_eval],  # 评估训练集
                    feature_name=feature_name,
                    verbose_eval = 50,
                    num_boost_round = num_tree,
                   early_stopping_rounds = 20,
                   callbacks=[lgb.record_evaluation(eval_result)])

    if print_plot == True:
        # plot loss
        plt.title('train_loss')
        for data_name, metric_res in eval_result.items():
            for metric_name, log_ in metric_res.items():
                plt.plot(log_, label = f'{data_name}-{metric_name}', 
                        color='steelblue' if 'train' in data_name else 'darkred', 
                        linestyle=None if 'train' in data_name else '-.',
                        alpha=0.7)

    #train precision recall and accuracy
    y_train_pred = gbm.predict(X_train)>threshold
    print(f'The train accuracy is {sum(y_train_pred == y_train)/len(y_train_pred)}')

    precision = precision_score(y_train, y_train_pred, average = 'binary')
    print(f'The train precision is {precision}')

    recall = recall_score(y_train, y_train_pred, average = 'binary')
    print(f'The train recall is {recall}')

    auc = roc_auc_score(y_train, gbm.predict(X_train))
    print(f'The training auc is {auc}')


    #test precision recall and accuracy
    y_test_pred = gbm.predict(X_test)>threshold
    print(f'The test accuracy is {sum(y_test_pred == y_test)/len(y_test_pred)}')

    precision = precision_score(y_test, y_test_pred, average = 'binary')
    print(f'The test precision is {precision}')

    recall = recall_score(y_test, y_test_pred, average = 'binary')
    print(f'The test recall is {recall}')

    auc = roc_auc_score(y_test, gbm.predict(X_test))
    print(f'The testing auc is {auc}')

    if return_pred ==True:
        return gbm, gbm.predict(X_test)
    return gbm


# In[84]:


def feature_importance(gbm, X, return_val = False):
    importance = gbm.feature_importance()
    feature_names = np.array(X.columns)

    # Sort feature importances in descending order
    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = feature_names[sorted_idx]
    sorted_importance = importance[sorted_idx]

    # Display feature importances
    for name, importance in zip(sorted_names, sorted_importance):
        print(f"{name}: {importance}")
    if return_val ==True:
        return sorted_names, sorted_importance
    return 

def feature_imp(gbm, X, return_val = False):
    importance = gbm.feature_importance()
    feature_names = np.array(X.columns)
    return feature_names, importance

def sharpley_value(model, X_test, X, importance_plot = False, shap_force_plot = False, num_feature_display=None):
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    if shap_force_plot == True:
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values[1], X_test[1])

    if importance_plot == True:
        if num_feature_display:
            shap.summary_plot(shap_values, X_test, feature_names = X.columns, max_display = num_feature_display)
        else:
            shap.summary_plot(shap_values, X_test, feature_names = X.columns)
    # Plot the feature importance using SHAP summary plot
    if num_feature_display:
        shap.summary_plot(shap_values[1], X_test ,feature_names = X.columns, max_display = num_feature_display)
    else:
        shap.summary_plot(shap_values[1], X_test ,feature_names = X.columns)

    return explainer, shap_values



# In[13]:


def select_threshold_val(X_train, y_train, X_test, y_test, model):
    #define threshold value which above it will be classified as 1
    threshold = np.arange(0.2,0.51,0.01)

    f1_results = []

    for threshold_val in threshold:
        #train precision recall and accuracy
        y_train_pred = model.predict(X_train)>threshold_val

        #select the highest f score threshold from train data
        f1_results.append(f1_score(y_train, y_train_pred, average = 'binary'))

    max_ind = f1_results.index(max(f1_results))

    #return threshold with highest f1 score
    print(f'The highest threshold value is {threshold[max_ind]}, with f1 score {f1_results[max_ind]}.')

    #test precision recall and accuracy
    y_test_pred = model.predict(X_test)>threshold[max_ind]
    print(f'The test accuracy is {sum(y_test_pred == y_test)/len(y_test_pred)}')

    precision = precision_score(y_test, y_test_pred, average = 'binary')
    print(f'The test precision is {precision}')

    recall = recall_score(y_test, y_test_pred, average = 'binary')
    print(f'The test recall is {recall}')


    return 




# ## 1.  100天可召回作为y值

# In[51]:


#import table
from nbsdk import get_table, get_pandas
table_all = get_table('8c634b8e-674c-4093-add7-8c879ed6f4a9/sx_churn_feature_larger.table')
df_all = table_all.to_pandas()

X_tmp_return,y_return = df_all.iloc[:,:-4], df_all.iloc[:,-2]
X_return = pd.get_dummies(X_tmp_return)
X_train_return, X_test_return, y_train_return, y_test_return = train_test_split(X_return,y_return,test_size = 0.25, random_state = 1)
scaler = preprocessing.StandardScaler().fit(X_train_return)
X_train_return = scaler.transform(X_train_return)
X_test_return = scaler.transform(X_test_return)


# In[79]:


#CV调参
cv_res_return = cross_val_gbm(X_train_return, y_train_return)


# In[94]:


#根据调参训练模型
opt_pair_return=best_parameter(cv_res_return)
gbm_return = lgb_train_function(X_train_return, y_train_return, X_test_return, y_test_return, learning_rate = opt_pair_return[0], max_depth_tree = opt_pair_return[1],num_tree = 500)
feature_importance(gbm_return, X_return)


# In[91]:


#改变threshold值
select_threshold_val(X_train_return, y_train_return, X_test_return, y_test_return,gbm_return)


# In[58]:


#特征重要性
sharpley_value(gbm_return, X_test_return, X_return,importance_plot = True)


# ## 2.  上次订单100天前作为y值

# In[43]:


X_tmp_100_nonrecall,y_100_nonrecall = df_all.iloc[:,:-4], df_all.iloc[:,-4]
X_100_nonrecall = pd.get_dummies(X_tmp_100_nonrecall)
X_train_100_nonrecall, X_test_100_nonrecall, y_train_100_nonrecall, y_test_100_nonrecall = train_test_split(X_100_nonrecall,y_100_nonrecall,test_size = 0.25, random_state = 1)
scaler = preprocessing.StandardScaler().fit(X_train_100_nonrecall)
X_train_100_nonrecall = scaler.transform(X_train_100_nonrecall)
X_test_100_nonrecall = scaler.transform(X_test_100_nonrecall)


# In[44]:


cv_res_100_nonrecall = cross_val_gbm(X_train_100_nonrecall, y_train_100_nonrecall)


# In[95]:


#根据调参训练模型
opt_pair_100_nonrecall=best_parameter(cv_res_100_nonrecall)
gbm_100_nonrecall = lgb_train_function(X_train_100_nonrecall, y_train_100_nonrecall, X_test_100_nonrecall, y_test_100_nonrecall, learning_rate = opt_pair_100_nonrecall[0], max_depth_tree = opt_pair_100_nonrecall[1],num_tree = 500)
feature_importance(gbm_100_nonrecall, X_100_nonrecall)


# In[ ]:


#特征重要性
sharpley_value(gbm_100_nonrecall, X_test_100_nonrecall, X_100_nonrecall, importance_plot = True)


# In[ ]:





# ## 3.  100天可召回作为y值 oversample可召回人数

# In[47]:


from nbsdk import get_table, get_pandas
oversample_table = get_table('b0815750-eb27-431d-9c8e-053581787e25/sx_oversampled_feature_tmp.table')
df_oversample = oversample_table.to_pandas()


X_tmp_oversample,y_oversample = df_oversample.iloc[:,:-4], df_oversample.iloc[:,-2]
X_oversample = pd.get_dummies(X_tmp_oversample)
X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample = train_test_split(X_oversample,y_oversample,test_size = 0.25, random_state = 1)
scaler = preprocessing.StandardScaler().fit(X_train_oversample)
X_train_oversample = scaler.transform(X_train_oversample)
X_test_oversample = scaler.transform(X_test_oversample)


# In[48]:


cv_res_oversample = cross_val_gbm(X_train_oversample, y_train_oversample)


# In[96]:


#根据调参训练模型
opt_pair_oversample=best_parameter(cv_res_oversample)
gbm_100_oversample = lgb_train_function(X_train_oversample, y_train_oversample, X_test_oversample, y_test_oversample, learning_rate = opt_pair_oversample[0], max_depth_tree = opt_pair_oversample[1],num_tree = 500,threshold = 0.5)
feature_importance(gbm_100_oversample, X_100_nonrecall)



# In[92]:


lgb_train_function(X_train_oversample, y_train_oversample, X_test_oversample, y_test_oversample, learning_rate = 0.05, max_depth_tree = None,num_tree = 500,threshold = 0.5)


# In[60]:


#特征重要性
sharpley_value(gbm_100_oversample, X_test_oversample, X_oversample,importance_plot=True)


# ## 5. 加入券信息后

# from nbsdk import get_table, get_pandas
# table_new = get_table('e92229cd-c6a7-4389-8d31-31d3d9970478/churn_feat_diff_final.table')
# df_new = table_new.to_pandas()
# 

# In[427]:



from nbsdk import get_table, get_pandas
table_new = get_table('9736d27e-f08a-49f1-9a6d-63da88270d7b/sx_reduce_churn_exploration_3.table')
df_new = table_new.to_pandas()
df_new.loc[:,"ta_diff"]=df_new.loc[:,"ta_diff"].astype("float")
df_new.loc[:,"tc_diff"]=df_new.loc[:,"tc_diff"].astype("float")

X_tmp_100_new,y_100_new = df_new.iloc[:,1:-4], df_new.iloc[:,-4]
X_100_new = pd.get_dummies(X_tmp_100_new)
X_train_before_100_new, X_test_before_100_new, y_train_100_new, y_test_100_new = train_test_split(X_100_new,y_100_new,test_size = 0.25, random_state = 1)
scaler = preprocessing.StandardScaler().fit(X_train_before_100_new)
X_train_100_new = scaler.transform(X_train_before_100_new)
X_test_100_new = scaler.transform(X_test_before_100_new)


# In[144]:


continuous_df = df_new.select_dtypes(include=[int, "int32", float])
cor_mat = continuous_df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(cor_mat, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(len(cor_mat.columns)), cor_mat.columns, rotation=45)
plt.yticks(np.arange(len(cor_mat.columns)), cor_mat.columns)
plt.title('Correlation Plot')
plt.show()


# In[146]:


cv_res_new = cross_val_gbm(X_train_100_new, y_train_100_new)


# In[228]:


#opt_pair_new=best_parameter(cv_res_new)
gbm_100_new = lgb_train_function(X_train_100_new, y_train_100_new, X_test_100_new, y_test_100_new, learning_rate = 0.05, max_depth_tree = None,num_tree = 500,print_plot = True,threshold = 0.5)

#gbm_100_new = lgb_train_function(X_train_100_new, y_train_100_new, X_test_100_new, y_test_100_new, learning_rate = opt_pair_new[0], max_depth_tree = opt_pair_new[1], num_tree = 500,print_plot = True,threshold = 0.5)
feature_importance(gbm_100_new, X_100_new)


# In[229]:


explainer_new, shap_values_new = sharpley_value(gbm_100_new, X_test_100_new, X_100_new, importance_plot = True)

def shap_dep_plot_new(feature_name,largest_scale, smallest_scale=None, X = X_100_new, X_test = np.array(X_test_before_100_new.fillna(-1)), shap_values = shap_values_new):
    ta_ind = list(np.where(X.columns==feature_name))[0][0]
    if smallest_scale:
        select_ind = np.where((X_test[:,ta_ind]<largest_scale)&(X_test[:,ta_ind]>smallest_scale))
    else: 
        select_ind = np.where(X_test[:,ta_ind]<largest_scale)
    plt.scatter(X_test[:,ta_ind][select_ind], shap_values[1][:,ta_ind][select_ind],s=10, alpha=1)
    plt.xlabel(f'{feature_name} value')
    plt.ylabel(f'{feature_name} shap')
    plt.show()

shap_dep_plot_new("ta_diff",500,-500)
shap_dep_plot_new("tc_diff",20,-20)
shap_dep_plot_new("tc",30,0)
shap_dep_plot_new("ta",750)
shap_dep_plot_new("delivery_ta",300)
shap_dep_plot_new("delivery_tc",50)
shap_dep_plot_new("distinct_store",15)
shap_dep_plot_new("avg_party_size",10)
shap_dep_plot_new("max_discount",500)
shap_dep_plot_new("min_discount",200,-50)

shap_dep_plot_new("all_num_coupon",100)
shap_dep_plot_new("ai_num_coupon",100)

shap_dep_plot_new("all_num_redeem",40)
shap_dep_plot_new("ai_num_redeem",40)

shap_dep_plot_new("ai_redeemed_avg_party_size",4,1)
shap_dep_plot_new("all_redeemed_avg_party_size",4,1)


# In[264]:


# shap 绝对值排序
#shap_abs_avg = [np.sum(abs(shap_values_new[1][:,i]))/len(shap_values_new[1][:,i]) for i in range(len(shap_values_new[1][0,:]))]
average_shap_values = abs(shap_values_new[1]).mean(axis=0)

shap_abs = {
    'variable_name':X_100_new.columns,
    'shap_val':average_shap_values
}

df_shap_abs = pd.DataFrame(shap_abs)
df_shap_abs = df_shap_abs.sort_values(by='shap_val', ascending = False)
df_shap_abs['ranking'] = list(range(1,len(average_shap_values)+1))
pd.set_option('display.max_rows', None)
df_shap_abs


# In[309]:


#有ok餐平均shap
ok_can_ind = list(np.where(X_100_new.columns=="if_ok_can"))[0][0]
select_ind_equal_one = np.where(X_test_before_100_new.iloc[:,ok_can_ind]==1)
shap_values_new[1][:,ok_can_ind][select_ind_equal_one]



# In[318]:


data = {
        'Group': np.array(X_test_before_100_new.fillna(-1))[:,ok_can_ind],
        'Values': shap_values_new[1][:,ok_can_ind]
    }

df = pd.DataFrame(data)

# Create a dictionary to store the values for each group
group_data = {}
for group, values in df.groupby('Group')['Values']:
    group_data[group] = values.tolist()

# Convert the dictionary values to a list for boxplot
boxplot_data = list(group_data.values())

# Create the box plot
#plt.boxplot(boxplot_data, labels=list(group_data.keys()), sym='')
plt.boxplot(boxplot_data, labels=list(group_data.keys()))


# 
# shap_dep_plot_new("num_sanfang",10)
# shap_dep_plot_new("douyin_redeem",10)
# shap_dep_plot_new("meituan_redeem",20)
# shap_dep_plot_new("num_douyin",30)
# shap_dep_plot_new("num_meituan",30)
# 
# 
# 

# In[239]:


shap_dep_plot_new("working_trade_zone",1)
shap_dep_plot_new("traditional_business_district",1)
shap_dep_plot_new("university",1)
shap_dep_plot_new("shopping_center",1)
shap_dep_plot_new("traffic_center",1)
shap_dep_plot_new("travel_trade_zone",1)
shap_dep_plot_new("shopping_mall",1)
shap_dep_plot_new("other_business_district",1)
shap_dep_plot_new("social_business_district",1)
shap_dep_plot_new("city_public_traffic",1)
shap_dep_plot_new("hospital",1)
shap_dep_plot_new("market_trade_zone",1)
shap_dep_plot_new("high_app_wechat_tc_percent",1)
shap_dep_plot_new("douyin_redeem_percent",1)
shap_dep_plot_new("meituan_redeem_percent",1)
shap_dep_plot_new("sanfang_percent",1)
shap_dep_plot_new("app_tc_percent",1)
shap_dep_plot_new("wechat_tc_percent",1)
shap_dep_plot_new("num_app_tc",15)
shap_dep_plot_new("num_wechat_tc",15)
shap_dep_plot_new("beef_burger_tc",15)
shap_dep_plot_new("whole_chicken_tc",15)
shap_dep_plot_new("coffee_tc",15)
shap_dep_plot_new("ok_can_tc",15)
shap_dep_plot_new("num_beef_burger",15)
shap_dep_plot_new("num_whole_chicken",15)
shap_dep_plot_new("num_coffee_tc",15)
shap_dep_plot_new("num_ok_can_tc",15)


# In[173]:


X_100_new.columns[160:200]


# In[319]:


feat_ind_1 = list(np.where(X_100_new.columns=="if_ok_can"))[0][0]
feat_ind_1
data = {
        'Group': np.array(X_test_before_100_new.fillna(-1))[:,feat_ind_1],
        'Values': shap_values_new[1][:,feat_ind_1]
    }
df = pd.DataFrame(data)

# Create a dictionary to store the values for each group
group_data = {}
for group, values in df.groupby('Group')['Values']:
    group_data[group] = values.tolist()

# Convert the dictionary values to a list for boxplot
boxplot_data = list(group_data.values())

# Create the box plot
#plt.boxplot(boxplot_data, labels=list(group_data.keys()), sym='')
plt.boxplot(boxplot_data, labels=list(group_data.keys()))


# In[321]:


import matplotlib.pyplot as plt
def box_plot(feature_name,X = X_100_new, X_test = np.array(X_test_before_100_new.fillna(-1)), shap_values = shap_values_new):    
    feat_ind = list(np.where(X.columns==feature_name))[0][0]
    # Sample data with two columns: group and values
    data = {
        'Group': X_test[:,feat_ind],
        'Values': shap_values[1][:,feat_ind]
    }
    print(feat_ind)

    df = pd.DataFrame(data)

    # Create a dictionary to store the values for each group
    group_data = {}
    for group, values in df.groupby('Group')['Values']:
        group_data[group] = values.tolist()

    # Convert the dictionary values to a list for boxplot
    boxplot_data = list(group_data.values())

    # Create the box plot
    #plt.boxplot(boxplot_data, labels=list(group_data.keys()), sym='')
    plt.boxplot(boxplot_data, labels=list(group_data.keys()))

    # Set axis labels and title
    plt.xlabel('Group')
    plt.ylabel('Values')
    plt.title(f'Box Plot by {feature_name}')

    # Show the plot
    plt.show()

box_plot("his_family_flag_Y")
box_plot("his_travel_flag_Y")
box_plot("student_flag_Y")
box_plot("his_value_seeker_flag_Y")
box_plot("his_delivery_flag_Y")
box_plot("his_lto_flag_Y")
box_plot("his_working_day_lunch_flag_Y")
box_plot("his_breakfast_flag_Y")
box_plot("his_afternoon_flag_Y")
box_plot("his_dinner_flag_Y")
box_plot("his_coffee_flag_Y")
box_plot("elderly_flag_Y")



# In[325]:


box_plot("if_beef_burger")
box_plot("if_whole_chicken")
box_plot("if_coffee")
box_plot("if_ok_can")


# In[234]:


def calculate_woe(df):
    a = np.sum((df['churn']==1) & (df['feature']==1))
    b = np.sum((df['churn']==0) & (df['feature']==1))
    c = np.sum((df['churn']==1) & (df['feature']==0))
    d = np.sum((df['churn']==0) & (df['feature']==0))

    return np.log(a/(a+c)/b*(b+d))+np.log(c/(a+c)/d*(b+d))


def woe_feat(feature_name, X = X_100_new, y_test = y_100_new):

    # Example usage:
    data = {
        'feature': list(X.loc[:,feature_name]),
        'churn': y_test
    }

    df = pd.DataFrame(data)
    transformed_df = calculate_woe(df)
    print(transformed_df)

print(f"woe value for his family_flag is {woe_feat('his_family_flag_Y')}")
print(f'woe value for his_travel_flag_Y is {woe_feat("his_travel_flag_Y")}')
print(f'woe value for student_flag_Y is {woe_feat("student_flag_Y")}')
print(f'woe value for his_value_seeker_flag_Y is {woe_feat("his_value_seeker_flag_Y")}')
print(f'woe value for his_delivery_flag_Y is {woe_feat("his_delivery_flag_Y")}')
print(f'woe value for his_lto_flag_Y is {woe_feat("his_lto_flag_Y")}')
print(f'woe value for his_working_day_lunch_flag_Y is {woe_feat("his_working_day_lunch_flag_Y")}')
print(f'woe value for his_breakfast_flag_Y is {woe_feat("his_breakfast_flag_Y")}')
print(f'woe value for his_afternoon_flag_Y is {woe_feat("his_afternoon_flag_Y")}')
print(f'woe value for his_dinner_flag_Y is {woe_feat("his_dinner_flag_Y")}')
print(f'woe value for his_coffee_flag_Y is {woe_feat("his_coffee_flag_Y")}')
print(f'woe value for elderly_flag_Y is {woe_feat("elderly_flag_Y")}')


# In[327]:


print(f'woe value for if_beef_burger is {woe_feat("if_beef_burger")}')
print(f'woe value for if_whole_chicken is {woe_feat("if_whole_chicken")}')
print(f'woe value for if_coffee is {woe_feat("if_coffee")}')
print(f'woe value for if_ok_can is {woe_feat("if_ok_can")}')




# In[195]:


X_100_new.iloc[:,-1]


# In[ ]:





# In[58]:


X_test = np.array(X_test_before_100_new.fillna(-1))
vs_ind = list(np.where(X_100_new.columns=="his_value_seeker_flag_Y"))[0][0]
redeem_ind = list(np.where(X_100_new.columns=="num_redeem"))[0][0]

select_ind =  np.where((X_test[:,vs_ind]==0)&(X_test[:,redeem_ind]>10)&(shap_values_new[1][:,redeem_ind]>0.6))


shap.initjs()
shap.force_plot(explainer_new.expected_value[1], shap_values_new[1][select_ind[0][0],:], X_test[select_ind[0][0],:], feature_names = X_100_new.columns)


# In[59]:


shap.force_plot(explainer_new.expected_value[1], shap_values_new[1][select_ind[0][3],:], X_test[select_ind[0][3],:], feature_names = X_100_new.columns)


# In[80]:


df_diff


# In[81]:


data_column1


# In[86]:


df_diff


# In[88]:



from nbsdk import get_table, get_pandas
table_diff = get_table('5b8e145e-d337-42ee-9e57-c181b6486b11/ta_tc_diff.table')
df_diff = table_diff.to_pandas()


import matplotlib.pyplot as plt
import numpy as np


# Sample data
data_column1 = df_diff.iloc[0,1:]
data_column2 = df_diff.iloc[1,1:]

# Calculate the difference between the two columns
data_difference = np.array(data_column1) - np.array(data_column2)

# Create the figure and axes
fig, ax1 = plt.subplots(figsize=(10,6))

# Plot the two columns as bars
bar_width = 0.35
x = np.arange(len(data_column1))
bar1 = ax1.bar(x, data_column1, width=bar_width, label='increase sales > 200',color='blue', alpha=0.7)
bar2 = ax1.bar(x + bar_width, data_column2, width=bar_width, label='decrease sales > 200', color='red', alpha=0.7)



# Set labels for axes and title
ax1.set_xlabel('Month')
ax1.set_ylabel('Percent User')
ax1.set_title('Differ in sales among different month')

# Set x-axis tick labels
ax1.set_xticks(x + bar_width / 2)
ax1.set_xticklabels(['7', '8', '9', '10', '11','12','1','2','3','4','5','6'])

# Create the second Y-axis (ax2) for the line plot
ax2 = ax1.twinx()

# Plot the line specifying the difference between the two columns
ax2.plot(x + bar_width/2, data_difference, label='Difference', color='green', marker='o')

# Set label for the second Y-axis
ax2.set_ylabel('Difference', color='green')

# Show grid lines for both Y-axes
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Show legend for both plots
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.show()


# In[140]:


gbm_100_new, y_pred = lgb_train_function(X_train_100_new, y_train_100_new, X_test_100_new, y_test_100_new, learning_rate = 0.05, max_depth_tree = None,num_tree = 500,return_pred = True,print_plot = True,threshold = 0.5)


# ta_diff 和tc_diff都高
# 

# In[106]:


X_test = np.array(X_test_before_100_new.fillna(-1))
vs_ind = list(np.where(X_100_new.columns=="ta_diff"))[0][0]
redeem_ind = list(np.where(X_100_new.columns=="tc_diff"))[0][0]

select_ind =  np.where((X_test[:,vs_ind]>200)&(X_test[:,redeem_ind]>10)&(shap_values_new[1][:,redeem_ind]>0.1))

select_ind
#shap.initjs()
shap.force_plot(explainer_new.expected_value[1], shap_values_new[1][select_ind[0][0],:], X_test[select_ind[0][0],:], feature_names = X_100_new.columns)


# In[123]:


X_test = np.array(X_test_before_100_new.fillna(-1))
vs_ind = list(np.where(X_100_new.columns=="ta_diff"))[0][0]
redeem_ind = list(np.where(X_100_new.columns=="tc_diff"))[0][0]

select_ind =  np.where((X_test[:,vs_ind]>500))
#shap.initjs()
print(f'tc diff is {X_test[select_ind[0][0],redeem_ind]}')
shap.force_plot(explainer_new.expected_value[1], shap_values_new[1][select_ind[0][0],:], X_test[select_ind[0][0],:], feature_names = X_100_new.columns)


# ## 可视化：
# 

# In[533]:


#!source activate python37 && pip3 install typing_extensions -i https://pypi.douban.com/simple
#!pip3 install altair vega_datasets
#!pip3 freeze
#!pip3 install typing_extensions==4.4.0
#!source activate python37 && pip3 install vega -i https://pypi.douban.com/simple
#!source activate python37 && pip3 install altair vega_datasets -i https://pypi.douban.com/simple


# In[603]:


get_ipython().system(u'source activate python37 && pip3 install vega==1.4.0 -i https://pypi.douban.com/simple')
get_ipython().system(u'pip3 install vega==1.4.0')


# In[599]:


#!pip install vega
#!pip3 install --upgrade notebook
get_ipython().system(u'jupyter nbextension install --sys-prefix --py vega')
#!jupyter nbextension enable vega --py --sys-prefix


# In[604]:


get_ipython().system(u'pip freeze')


# 解决无法import self from typing_extensions
# import typing_extensions
# from importlib import reload
# reload(typing_extensions)

# In[605]:


import altair as alt
from vega_datasets import data
import vega
reload(vega)


# In[606]:


alt.renderers.enable('notebook')


# In[689]:


from vega_datasets import data
cars = data.cars()

a = alt.Chart(cars).mark_point().encode(
    x = 'Horsepower',
    y = 'Miles_per_Gallon',
    color = 'Origin')
#a


# In[456]:


#要convert to character试
feat_ind = list(np.where(X_100_new.columns=="num_wechat_tc"))[0][0]
select_ind =  np.where((X_test_before_100_new.iloc[:,feat_ind]<=10))[0]
# Sample data with two columns: group and values
data = {
    'Month': np.array(X_test_before_100_new.iloc[:,feat_ind])[select_ind],
    'temp_max': shap_values_new[1][:,feat_ind][select_ind]
}
#print(feat_ind)

df = pd.DataFrame(data)
df.to_csv("num_wechat_tc.csv")


# In[685]:


a = pd.read_csv("num_wechat_tc.csv")
print(list(a['temp_max'])[1:1000])


# In[686]:


step = 20
overlap = 1

alt.Chart(source, height=step).transform_timeunit(
    Month='month(date)'
).transform_joinaggregate(
    mean_temp='mean(temp_max)', groupby=['Month']
).transform_bin(
    ['bin_max', 'bin_min'], 'temp_max'
).transform_aggregate(
    value='count()', groupby=['Month', 'mean_temp', 'bin_min', 'bin_max']
).transform_impute(
    impute='value', groupby=['Month', 'mean_temp'], key='bin_min', value=0
).mark_area(
    interpolate='monotone',
    fillOpacity=0.8,
    stroke='lightgray',
    strokeWidth=0.5
).encode(
    alt.X('bin_min:Q')
        .bin(True)
        .title('Maximum Daily Temperature (C)'),
    alt.Y('value:Q')
        .axis(None)
        .scale(range=[step, -step * overlap]),
    alt.Fill('mean_temp:Q')
        .legend(None)
        .scale(domain=[30, 5], scheme='redyellowblue')
).facet(
    alt.Row('Month:T')
        .title(None)
        .header(labelAngle=0, labelAlign='right', format='%B')
).properties(
    title='Seattle Weather',
    bounds='flush'
).configure_facet(
    spacing=0
).configure_view(
    stroke=None
).configure_title(
    anchor='end'
)


# In[493]:


step = 20
overlap = 1

alt.Chart("num_wechat_tc.csv", height=step).transform_joinaggregate(
    mean_temp='mean(temp_max)', groupby=['Month']
).transform_bin(
    ['bin_max', 'bin_min'], 'temp_max'
).transform_aggregate(
    value='count()', groupby=['Month', 'mean_temp', 'bin_min', 'bin_max']
).transform_impute(
    impute='value', groupby=['Month', 'mean_temp'], key='bin_min', value=0
).mark_area(
    interpolate='monotone',
    fillOpacity=0.8,
    stroke='lightgray',
    strokeWidth=0.5
).encode(
    alt.X('bin_min:Q')
        .bin(True)
        .title('Maximum Daily Temperature (C)'),
    alt.Y('value:Q')
        .axis(None)
        .scale(range=[step, -step * overlap]),
    alt.Fill('mean_temp:Q')
        .legend(None)
        .scale(domain=[30, 5], scheme='redyellowblue')
).facet(
    alt.Row('Month:T')
        .title(None)
        .header(labelAngle=0, labelAlign='right', format='%B')
).properties(
    title='Seattle Weather',
    bounds='flush'
).configure_facet(
    spacing=0
).configure_view(
    stroke=None
).configure_title(
    anchor='end'
)




# In[ ]:



alt.Chart("num_wechat_tc.csv", height=step).transform_joinaggregate(
    mean_temp='mean(temp_max)', groupby=['Month']
).transform_bin(
    ['bin_max', 'bin_min'], 'temp_max'
).transform_aggregate(
    value='count()', groupby=['Month', 'mean_temp', 'bin_min', 'bin_max']
).transform_impute(
    impute='value', groupby=['Month', 'mean_temp'], key='bin_min', value=0
).mark_area(
    interpolate='monotone',
    fillOpacity=0.8,
    stroke='lightgray',
    strokeWidth=0.5
).encode(
    alt.X('bin_min:Q')
        .bin(True)
        .title('Maximum Daily Temperature (C)'),
    alt.Y('value:Q')
        .axis(None)
        .scale(range=[step, -step * overlap]),
    alt.Fill('mean_temp:Q')
        .legend(None)
        .scale(domain=[30, 5], scheme='redyellowblue')
).facet(
    alt.Row('Month:T')
        .title(None)
        .header(labelAngle=0, labelAlign='right', format='%B')
).properties(
    title='Seattle Weather',
    bounds='flush'
).configure_facet(
    spacing=0
).configure_view(
    stroke=None
).configure_title(
    anchor='end'
)


# In[478]:


np.array(X_test_before_100_new.iloc[:,feat_ind])[select_ind]


# In[671]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ridge_plot(feature_name):
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


        feat_ind = list(np.where(X_100_new.columns==feature_name))[0][0]
        select_ind =  np.where((X_test_before_100_new.iloc[:,feat_ind]<=20))[0]

        # Create the data
        df = pd.DataFrame(dict(x=shap_values_new[1][:,feat_ind][select_ind], g=np.array(X_test_before_100_new.iloc[:,feat_ind])[select_ind]))


        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x", color = '#CD2626',
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "x", clip_on=False, color="r", lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color="grey", clip_on=False)


        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color='grey',
                    ha="left", va="center", transform=ax.transAxes)

            #ax.set_xlim(left=-0.6, right=0.2)

        g.map(label, "x")

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.set(xlabel = "shap value")
        g.despine(bottom=True, left=True)

       # mean_values = df.groupby("g")["x"].mean().reset_index()
       # for g_val in range(len(mean_values)):
        #    mean_value = mean_values.iloc[g_val,:]
        #    g.axes.flat[0].axvline(x=mean_value, color = 'red',linestyle='--')

ridge_plot("num_wechat_tc")


# In[672]:


ridge_plot("num_app_tc")
ridge_plot("beef_burger_tc")
ridge_plot("whole_chicken_tc")
ridge_plot("coffee_tc")
ridge_plot("ok_can_tc")


# In[670]:


#均值

def plot_mean(feature_name):
    feat_ind = list(np.where(X_100_new.columns==feature_name))[0][0]
    select_ind =  np.where((X_test_before_100_new.iloc[:,feat_ind]<=20))[0]

    # Create the data
    df = pd.DataFrame(dict(x=shap_values_new[1][:,feat_ind][select_ind], g=np.array(X_test_before_100_new.iloc[:,feat_ind])[select_ind]))
    mean_values = df.groupby("g")["x"].mean().reset_index()

    plt.figure(figsize=(8,6))
    sns.set_style('whitegrid')

    sns.lineplot(data = mean_values, x = "g", y='x',color = '#CD2626')
    sns.scatterplot(data = mean_values, x = "g", y='x', marker = 'o',color = 'grey')
    plt.xlabel("tc")
    plt.ylabel("shap")
    plt.title(f"mean shap value over {feature_name}")
    plt.show()
plot_mean("num_wechat_tc")
plot_mean("num_app_tc")
plot_mean("beef_burger_tc")
plot_mean("whole_chicken_tc")
plot_mean("coffee_tc")
plot_mean("ok_can_tc")


# In[642]:







# In[627]:


ridge_plot("num_app_tc")


# In[673]:


sns.set_style('whitegrid')
def regression_plot(feature_name):
    #define feature ind
    feat_ind = list(np.where(X_100_new.columns==feature_name))[0][0]
    select_ind =  np.where((X_test_before_100_new.iloc[:,feat_ind]>=0) & (X_test_before_100_new.iloc[:,feat_ind]<=1))[0]

    # Create the data
    df = pd.DataFrame(dict(x=shap_values_new[1][:,feat_ind][select_ind], g=np.array(X_test_before_100_new.iloc[:,feat_ind])[select_ind]))


    # regression plot with resizing ability: using plt subplots

    fig, ax = plt.subplots(figsize=(8, 8))

    sns.regplot(x='g', y='x', data=df, scatter=True, scatter_kws={'color':"grey"},line_kws={"color":'#CD2626'}, ax=ax)

    #plt.title("Regression plot for lwt and bwt")

    ax.set_title(f"Regression plot for {feature_name} and shap")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("shap value")
    plt.show()

regression_plot("douyin_redeem_percent")
regression_plot("meituan_redeem_percent")


# In[ ]: