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


# kmean clustering
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



#kmean with kval=4 cluster
kval = 4

#读hive文件
train_data_tmp = spark.sql("""select * from tmp.sx_reduce_churn_risk_exploration_data_new""")

feature_name_list = ['tc', 'ta', 'num_sanfang','distinct_store','ta_diff', 'if_coffee','avg_party_size','ai_num_coupon','all_num_coupon','ai_num_redeem','all_num_redeem','travel_flag','value_seeker_flag']
id_name_list = ['usercode_number']
train_data_t = train_data_tmp.filter(fn.col('churn_100') == 1)
t1 = train_data_t.withColumn("travel_flag", when(train_data_t.his_travel_flag == 'Y' ,1).when(train_data_t.his_travel_flag == 'N' ,0)).withColumn("value_seeker_flag", when(train_data_t.his_value_seeker_flag == 'Y' ,1).when(train_data_t.his_value_seeker_flag == 'N' ,0))
t1 = t1.dropDuplicates(['usercode_number'])











#Repeat the same thing but repeat k mean clustering 5 times to get the optimal clustering
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
