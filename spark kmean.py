# %%
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, HiveContext, SQLContext, functions as F

def get_spark_connection(queue="root.kp.xbr", instances=12, memory='32g', cores=4, partitions=100, **kwargs):
  sparkConf = SparkConf() \
    .setMaster("yarn") \
    .set("spark.yarn.queue", queue) \
    .set("spark.executor.memory", memory) \
    .set("spark.executor.cores", cores) \
    .set("spark.executor.instances", instances) \
    .set("spark.sql.shuffle.partitions", partitions)
  spark = SparkSession.builder.config(conf=sparkConf) \
   .appName("TestProforma") \
   .enableHiveSupport() \
   .getOrCreate()
  return spark

spark = get_spark_connection(queue='root.kp.xbr', instances=32,  cores=8, memmory='32G', partitions="100" )

# %%
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql.functions import when
from pyspark.ml.feature import  StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler,VectorAssembler, StringIndexer

# %%
train_data_tmp = spark.sql("""select * from tmp.sx_reduce_churn_risk_exploration_data_new order by rand() limit 10000""")

prev_data_name = ['tc', 'ta','num_sanfang', 'distinct_store','ta_diff', 'if_coffee','avg_party_size','ai_num_coupon','all_num_coupon','ai_num_redeem','all_num_redeem','churn_100','his_travel_flag']
data_name_list = ['tc', 'ta', 'num_sanfang','distinct_store','ta_diff', 'if_coffee','avg_party_size','ai_num_coupon','all_num_coupon','ai_num_redeem','all_num_redeem','travel_flag']



# %%
#只考虑低频人群去做kmean
train_data_t = train_data_tmp.select(prev_data_name).filter(fn.col('churn_100') == 1)
train_data = train_data_t.withColumn("travel_flag", when(train_data_t.his_travel_flag == 'Y' ,1).when(train_data_t.his_travel_flag == 'N' ,0)).select(data_name_list)



# %% [markdown]
# #k mean algorithm
# def k_mean_alg(data, k=5):
#     feature_columns = data.columns
#     assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features")
#     assembled_data = assembler.transform(data)
#     scaler = StandardScaler(inputCol='features', outputCol = 'scaled_features', withStd = True, withMean = True)
#     scaler_data = scaler.fit(assembled_data)
#     
#     #perform k mean clustering
#     kmeans = KMeans().setK(k).setSeed(1)
#     model = kmeans.fit(assembled_data)
#     
#     cluster_centers = model.clusterCenters()
#     predictions = model.transform(assembled_data)
#     return predictions
# 
# predictions = k_mean_alg(train_data,k=3)
#     

# %% [markdown]
# #standardize k mean algorithm
# def k_mean_alg(data, k=5):
#     feature_columns = data.columns
#     assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features_non_scale")
#     assembled_data = assembler.transform(data)
#     scaler = StandardScaler(inputCol='features_non_scale', outputCol = 'features', withStd = True, withMean = True)
#     scaler_model = scaler.fit(assembled_data)
#     scaler_df = scaler_model.transform(assembled_data)    
#     
#     #perform k mean clustering
#     kmeans = KMeans(k=k, seed = 10, featuresCol = 'features')
#     model = kmeans.fit(scaler_df)
#     
#     cluster_centers = model.clusterCenters()
#     predictions = model.transform(scaler_df)
#     return predictions
# 
# predictions = k_mean_alg(train_data,k=3)
#     

# %%
import random

#get the pseudo global minimum of withincluster sum of square distances from k mean algorithm
def k_mean_alg(data, k=5, num_randomize_round = 10):
    feature_columns = data.columns
    assembler = VectorAssembler(inputCols = feature_columns, outputCol = "features_non_scale")
    assembled_data = assembler.transform(data)
    scaler = StandardScaler(inputCol='features_non_scale', outputCol = 'features', withStd = True, withMean = True)
    scaler_model = scaler.fit(assembled_data)
    scaler_df = scaler_model.transform(assembled_data)    

    min_value = 1
    max_value = 10000
    random.seed(5)
    randomize_int = [random.randint(min_value, max_value) for i in range(num_randomize_round)]

    old_inertia = -100000
    for i in randomize_int:
        #perform k mean clustering
        kmeans = KMeans(k=k, seed = i, featuresCol = 'features')
        model = kmeans.fit(scaler_df)

        #compute within cluster sum of square distance
        new_inertia = model.computeCost(scaler_df)

        #if minimum then update optimum_model
        if new_inertia>old_inertia:
            optimum_model = model

        #update inertia
        old_inertia = new_inertia

    cluster_centers = optimum_model.clusterCenters()
    predictions = optimum_model.transform(scaler_df)
    return predictions

predictions = k_mean_alg(train_data,k=3)


# %% [markdown]
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# def visualization(predictions, feature_1, feature_2):
#     #get the predictions to pandas
#     pandas_df = predictions.select("prediction",feature_1,feature_2).toPandas()
#     
#     
#     #extract predictions
#     prediction_values = pandas_df["prediction"]
#     
#     #convert feature values that is suitable for ploting
#     x_values = pandas_df[feature_1]
#     y_values = pandas_df[feature_2]
#     
#     plt.scatter(x_values, y_values,c=prediction_values)
#     plt.xlabel(feature_1)
#     plt.ylabel(feature_2)
#     plt.title("K-means clustering result")
#     plt.show()
# 
# visualization(predictions,"ta_diff","tc")

# %%
import pandas as pd
import matplotlib.pyplot as plt

#ensure each label always have the same color
def visualization(predictions, feature_1, feature_2,f1_max_val = 10000, f2_max_val = 10000):
    #if f1_max_val:
        #get the predictions to pandas
    #   pandas_df = predictions.select("prediction",feature_1,feature_2).\
    #        filter(fn.col(feature_1)<f1_max_val).filter(fn.col(feature_2)<f2_max_val).toPandas()
    #else：
        #get the predictions to pandas
    #    pandas_df = predictions.select("prediction",feature_1,feature_2).toPandas()

    pandas_df = predictions.select("prediction",feature_1,feature_2).\
            filter(fn.col(feature_1)<f1_max_val).filter(fn.col(feature_2)<f2_max_val).toPandas()  

    #specify cluster colors 
    cluster_colors = {
        0:'red',
        1:'blue',
        2:'green',
        3:'yellow',
        4:"grey"
    }

    #extract predictions
    prediction_values = pandas_df["prediction"]


    #convert feature values that is suitable for ploting
    x_values = pandas_df[feature_1]
    y_values = pandas_df[feature_2]

    #count label 
    for cluster_id, group in pandas_df.groupby('prediction'):
        count_val = len(pandas_df[pandas_df["prediction"]==cluster_id])
        print(f'Group {cluster_id} has count {count_val} in color {cluster_colors.get(cluster_id, "black")}')


    plt.figure(figsize=(8,6))
    for cluster_id, group in pandas_df.groupby('prediction'):
        plt.scatter(group[feature_1],group[feature_2],color = cluster_colors.get(cluster_id, 'black'), label = f'Cluster {cluster_id}',alpha =.6)


    #plt.scatter(x_values, y_values,c=prediction_values)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.title("K-means clustering result")
    plt.show()

visualization(predictions,"ta_diff","tc",400,50)

# %%
visualization(predictions,"avg_party_size","tc",5,40)

# %%
#it seems like ta 
visualization(predictions,"ta","tc",1000, 50)

# %%
visualization(predictions,"ta","avg_party_size",1000,5)

# %%
visualization(predictions,"avg_party_size","distinct_store",5,20)

# %%
#低tc高收券人群，低tc低券人群，其他人群
visualization(predictions,"tc","all_num_coupon",50,60)


# %%
visualization(predictions,"all_num_redeem","all_num_coupon")


# %%
visualization(predictions,"distinct_store","tc",20,40)

# %%
visualization(predictions,"ta","all_num_coupon",1000,60)


# %%
visualization(predictions,"ai_num_coupon","all_num_coupon")


# %%
visualization(predictions,"all_num_coupon","all_num_redeem",60,8)


# %%
visualization(predictions,"num_sanfang","tc",12,50)


# %% [markdown]
# # 聚类分析：
# 

# %% [markdown]
# #spark读hdfs
# data_path = 'hdfs://yumcluster/kp/ks/workspace/PHApp_Recommend/pws/15/0ca574c8a7074d1abd8504d49d5125d5/726705/c125b861-c7b8-4c19-a527-462c6054c112/stages/10559155/0/data'
# df = spark.read.format('parquet').load(data_path)
# 
# #pyspark读hive
# from pyspark.sql import HiveContext
# sqlContext = HiveContext(sc)
# df = sqlContext.sql("Select * from aie_phapp_recommend.dim_content_feature_master limit 10")
# 
# #pyspark写hive
# df.write.format("hive").mode("overwrite").saveAsTable('tmp.hmh_spark2hive_test')
# 
# #聚类测试
# '''
# su - srv_kp_phapp_recomme
# /opt/spark2/bin/pyspark --master yarn --driver-memory 4G --executor-memory 16G --num-executors 8 --executor-cores 8 --queue root.kp.xbr
# '''
# 
# from pyspark.ml.feature import StandardScaler,VectorAssembler
# from pyspark.ml.clustering import KMeans
# from pyspark.ml import Pipeline
# 
# data_path = 'hdfs://yumcluster/kp/ks/workspace/PHApp_Recommend/telamon/15/ATM_explore/0925_aicluster_userprofile_2_table'
# t1 = spark.read.format('parquet').load(data_path)
# t1 = t1.filter('tc <= 100')
# 
# feature_name_list = ['after00_flag','after90_flag','after80_flag','after70_flag','before70_flag','tc','sales','ta','occasion_cnt','daypart_cnt','discount_rate','recency','breakfast_tc_percent','lunch_tc_percent','tea_tc_percent','dinner_tc_percent','night_tc_percent','carryout_tc_percent','delivery_tc_percent','dinein_tc_percent','tier1_tc_percent','tier2_tc_percent','tier3_tc_percent','tier456_tc_percent','party_size1_tc_percent','party_size2_tc_percent','party_size3_tc_percent','party_size4_tc_percent','party_size5above_tc_percent','workday_tc_percent','child_tc_percent','pizza_tc_percent','protein_tc_percent','rice_tc_percent','pasta_tc_percent','appetizer_tc_percent','drink_tc_percent']
# id_name_list = ['yumid']
# 
# vec_model = VectorAssembler(inputCols=feature_name_list, outputCol='raw_features')
# std_model = StandardScaler(withMean=True, withStd=True, inputCol='raw_features', outputCol='std_features')
# cls_model = KMeans(featuresCol='std_features', predictionCol='cls_label', k=25, maxIter=100)
# 
# whole_flow = Pipeline(stages=[vec_model, std_model, cls_model])
# 
# t1 = t1.dropDuplicates(['yumid']).dropna()
# t1 = whole_flow.fit(t1).transform(t1).select(id_name_list + feature_name_list + ['cls_label'])
# 
# agg_dict = {}
# for feature_name in feature_name_list:
#     agg_dict.update({feature_name:'avg'})
#   
# df_centers = t1.groupby('yumid').agg(agg_dict)
# 
# t1.write.format("hive").mode("overwrite").saveAsTable('tmp.hmh_spark2hive_test')

# %%
#读hive文件
train_data_tmp = spark.sql("""select * from tmp.sx_reduce_churn_risk_exploration_data_new""")

feature_name_list = ['tc', 'ta', 'num_sanfang','distinct_store','ta_diff', 'if_coffee','avg_party_size','ai_num_coupon','all_num_coupon','ai_num_redeem','all_num_redeem','travel_flag','value_seeker_flag']
id_name_list = ['usercode_number']
train_data_t = train_data_tmp.filter(fn.col('churn_100') == 1)
t1 = train_data_t.withColumn("travel_flag", when(train_data_t.his_travel_flag == 'Y' ,1).when(train_data_t.his_travel_flag == 'N' ,0)).withColumn("value_seeker_flag", when(train_data_t.his_value_seeker_flag == 'Y' ,1).when(train_data_t.his_value_seeker_flag == 'N' ,0))


#聚类测试
'''
su - srv_kp_phapp_recomme
/opt/spark2/bin/pyspark --master yarn --driver-memory 4G --executor-memory 16G --num-executors 8 --executor-cores 8 --queue root.kp.xbr
'''

from pyspark.ml.feature import StandardScaler,VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline


vec_model = VectorAssembler(inputCols=feature_name_list, outputCol='raw_features')
std_model = StandardScaler(withMean=True, withStd=True, inputCol='raw_features', outputCol='std_features')
cls_model = KMeans(featuresCol='std_features', predictionCol='cls_label', k=5, maxIter=100)

whole_flow = Pipeline(stages=[vec_model, std_model, cls_model])

t1 = t1.dropDuplicates(['usercode_number'])
t1 = whole_flow.fit(t1).transform(t1).select(id_name_list + feature_name_list + ['cls_label'])

agg_dict = {}
for feature_name in feature_name_list:
    agg_dict.update({feature_name:'avg'})

df_centers = t1.groupby('usercode_number').agg(agg_dict)

t1.write.format("hive").mode("overwrite").saveAsTable('tmp.sx_kmean_test_new')

# %%
t1.count()

# %%
#重复5次找出接近global minimum的kmeans

#kmean多少的cluster
kval = 4

#读hive文件
train_data_tmp = spark.sql("""select * from tmp.sx_reduce_churn_risk_exploration_data_new""")

feature_name_list = ['tc', 'ta', 'num_sanfang','distinct_store','ta_diff', 'if_coffee','avg_party_size','ai_num_coupon','all_num_coupon','ai_num_redeem','all_num_redeem','travel_flag','value_seeker_flag']
id_name_list = ['usercode_number']
train_data_t = train_data_tmp.filter(fn.col('churn_100') == 1)
t1 = train_data_t.withColumn("travel_flag", when(train_data_t.his_travel_flag == 'Y' ,1).when(train_data_t.his_travel_flag == 'N' ,0)).withColumn("value_seeker_flag", when(train_data_t.his_value_seeker_flag == 'Y' ,1).when(train_data_t.his_value_seeker_flag == 'N' ,0))
t1 = t1.dropDuplicates(['usercode_number'])


#聚类测试
'''
su - srv_kp_phapp_recomme
/opt/spark2/bin/pyspark --master yarn --driver-memory 4G --executor-memory 16G --num-executors 8 --executor-cores 8 --queue root.kp.xbr
'''

from pyspark.ml.feature import StandardScaler,VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline


vec_model = VectorAssembler(inputCols=feature_name_list, outputCol='raw_features')
std_model = StandardScaler(withMean=True, withStd=True, inputCol='raw_features', outputCol='std_features')
assembled_data = vec_model.transform(t1)
scaler_model = std_model.fit(assembled_data)
t1 = scaler_model.transform(assembled_data)   


old_inertia = -100000
for i in range(5):
    #没有设seed所以随机seed
    cls_model_tmp = KMeans(featuresCol='std_features', predictionCol='cls_label', k=kval, maxIter=100)

    #perform k mean clustering
    model = cls_model_tmp.fit(t1)

    #compute within cluster sum of square distance
    new_inertia = model.computeCost(t1)

    #if minimum then update optimum_model
    if new_inertia>old_inertia:
        optimum_model = model
    #update inertia
    old_inertia = new_inertia


t1 = optimum_model.transform(t1).select(id_name_list + feature_name_list + ['cls_label'])


agg_dict = {}
for feature_name in feature_name_list:
    agg_dict.update({feature_name:'avg'})

df_centers = t1.groupby('usercode_number').agg(agg_dict)

t1.write.format("hive").mode("overwrite").saveAsTable('tmp.sx_kmean_test_new')

# %%
t1.count()

# %%
df_centers

# %%