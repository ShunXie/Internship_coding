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
from pyspark.ml.feature import  StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StandardScaler,VectorAssembler, StringIndexer

# %%


# %%
train_data = spark.sql("""select * from aie_kfc_cltv.cltv_muti_label_traindata limit 10""") 

# %%
data_name_list = ['tc', 'ta', 'breakfast_tc', 'nonbreakfast_tc',\
'morning_tc', 'lunch_tc', 'afternoon_tc', 'dinner_tc', 'latenight_tc', \
'mon_tc', 'tue_tc', 'wen_tc', 'thu_tc', 'fri_tc', 'sat_tc', 'sun_tc',\
'tier1_tc', 'tier2_tc', 'tier3_tc', 'tier4_tc', 'tier5_tc', 'tier6_tc', \
'breakfast_maxta', 'nonbreakfast_maxta', 'morning_maxta', 'lunch_maxta',\
'afternoon_maxta', 'dinner_maxta', 'latenight_maxta', 'mon_maxta', 'tue_maxta', \
'wen_maxta', 'thu_maxta', 'fri_maxta', 'sat_maxta', 'sun_maxta', 'breakfast_minta', \
'nonbreakfast_minta', 'morning_minta', 'lunch_minta', 'afternoon_minta',\
'dinner_minta', 'latenight_minta', 'mon_minta', 'tue_minta', 'wen_minta',\
'thu_minta', 'fri_minta', 'sat_minta', 'sun_minta', 'breakfast_avgta', \
'nonbreakfast_avgta', 'morning_avgta', 'lunch_avgta', 'afternoon_avgta', \
'dinner_avgta', 'latenight_avgta', 'mon_avgta', 'tue_avgta', 'wen_avgta',\
'thu_avgta', 'fri_avgta', 'sat_avgta', 'sun_avgta', 'breakfast_sumta',\
'nonbreakfast_sumta', 'morning_sumta', 'lunch_sumta', 'afternoon_sumta', \
'dinner_sumta', 'latenight_sumta', 'mon_sumta', 'tue_sumta', 'wen_sumta', \
'thu_sumta', 'fri_sumta', 'sat_sumta', 'sun_sumta', 'breakfast_std_ta', \
'nonbreakfast_std_ta', 'morning_std_ta', 'lunch_std_ta', 'afternoon_std_ta', \
'dinner_std_ta', 'latenight_std_ta', 'mon_std_ta', 'tue_std_ta', 'wen_std_ta', \
'thu_std_ta', 'fri_std_ta', 'sat_std_ta', 'sun_std_ta', 'avg_discount', 'max_discount',\
'min_discount', 'sum_discount', 'std_discount', 'avg_ta_by_ps', 'std_ta_by_ps', \
'avg_city_tier', 'max_city_tier', 'min_city_tier', 'std_city_tier', 'avg_party_size',\
'max_party_size', 'min_party_size', 'std_party_size', 'cor_ta_da', 'distinct_daypart',\
'distinct_city', 'distinct_work_day', 'distinct_store', 'delivery_tc', 'delivery_ta', \
'avg_delivery_party_size', 'max_delivery_party_size', 'min_delivery_party_size',\
'std_delivery_party_size', 'preorder_tc', 'preorder_ta', 'avg_preorder_party_size', \
'max_preorder_party_size', 'min_preorder_party_size', 'std_preorder_party_size', \
'side_sold', 'coffee_sold', 'congee_sold', 'nutrition_sold', 'panini_sold', \
'riceroll_sold', 'dabing_sold', 'burger_sold', 'chickensnack_sold', 'cob_sold', \
'csd_sold', 'eggtart_sold', 'icecream_sold', 'sidefrenchfries_sold', \
'sideothers_sold', 'tea_sold', 'twister_sold', 'wing_sold', 'waffle_sold', \
'croissant_sold', 'nonfood_sold', 'pie_sold', 'juice_sold', 'rice_sold', 'lto_sold', \
'side_sell_price', 'coffee_sell_price', 'congee_sell_price', 'nutrition_sell_price',\
'panini_sell_price', 'riceroll_sell_price', 'dabing_sell_price', 'burger_sell_price', \
'chickensnack_sell_price', 'cob_sell_price', 'csd_sell_price', 'eggtart_sell_price', \
'icecream_sell_price', 'sidefrenchfries_sell_price', 'sideothers_sell_price', \
'tea_sell_price', 'twister_sell_price', 'wing_sell_price', 'waffle_sell_price', \
'croissant_sell_price', 'nonfood_sell_price', 'pie_sell_price', 'juice_sell_price', \
'rice_sell_price', 'lto_sell_price', 'single_product_number', 'combo_product_number', \
'kids_meal_toy_num', 'chicken_burger_soldratio_orlean', 'chicken_burger_soldratio_spicy', \
'chicken_burger_soldratio_crusty']



# %%
 #data perparation
label_list = ['re1']
train_data = spark.sql("""select * from aie_kfc_cltv.cltv_muti_label_traindata limit 10""") 
train_data = train_data.select(label_list + data_name_list)
#train_rdd = train_data.rdd.map(list)


vec_model = VectorAssembler(inputCols=data_name_list, outputCol='features')    

#model training
for i, label in enumerate(label_list):
    print(label)
    #trainSet = train_rdd.map(lambda x:Row(label=x[i], features=Vectors.dense(x[15:]))).toDF()
    #负采样 按正样本与负样本2:3的比例负采样
    pos_num = train_data.filter(fn.col(label) == 1).count()
    neg_num = train_data.filter(fn.col(label) == 0).count()
    prob = 1.5 * pos_num / (neg_num + 1e-6)
    trainSet = train_data.filter(fn.col(label) == 1).unionAll(train_data.filter(fn.col(label)==0).\
        withColumn('rand', fn.rand()).filter(fn.col('rand') <= prob).drop('rand'))
    #step size is learning rate
    gbdt = GBTClassifier(maxIter=250, maxDepth=10, featuresCol='features', labelCol=label,seed=42)

    whole_flow = Pipeline(stages=[vec_model, gbdt])
    gbdtModel = whole_flow.fit(trainSet)





# %%
# cv cross validation code
def cross_val(train_set):
    #step size is learning rate
    gbdt = GBTClassifier(maxIter=250, maxDepth=10, featuresCol='features', labelCol=label,seed=42)

    whole_flow = Pipeline(stages=[vec_model, gbdt])
    gbdtModel = whole_flow.fit(trainSet)

    #CV
    print("Start for cv")
    paramGrid = ParamGridBuilder() \
        .addGrid(gbdtModel.stepSize, [0.01,0.1]) \
        .build()

    # evaluate using binary classification auc
    evaluator = BinaryClassificationEvaluator(metricName = "areaUnderROC")
    whole_flow = Pipeline(stages=[vec_model, gbdt])

    # Create a CrossValidator with the pipeline, ParamGrid, and evaluator
    crossval = CrossValidator(estimator=whole_flow,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

    # Fit the CrossValidator to the data
    cvModel = crossval.fit(trainSet)

    # Get the best model
    best_pipeline = cvModel.bestModel

    print("End for cv")

    return best_pipeline




# %%
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator




def GBDT_cv():
     #data sampling
    label_list = ['re1']
    train_data = spark.sql("""select * from aie_kfc_cltv.cltv_muti_label_traindata limit 100""") 
    train_data = train_data.select(label_list + data_name_list)
    #train_rdd = train_data.rdd.map(list)

    vec_model = VectorAssembler(inputCols=data_name_list, outputCol='features')

    #model training 
    for i, label in enumerate(label_list):
        print(label)
        #trainSet = train_rdd.map(lambda x:Row(label=x[i], features=Vectors.dense(x[15:]))).toDF()
        #negative sampling 2:3
        pos_num = train_data.filter(fn.col(label) == 1).count()
        neg_num = train_data.filter(fn.col(label) == 0).count()
        prob = 1.5 * pos_num / (neg_num + 1e-6)
        trainSet = train_data.filter(fn.col(label) == 1).unionAll(train_data.filter(fn.col(label)==0).\
            withColumn('rand', fn.rand()).filter(fn.col('rand') <= prob).drop('rand'))

        trainSet_data_tmp = vec_model.transform(trainSet)
        trainSet_data = trainSet_data_tmp.withColumnRenamed("re1", "label") 

        #step size is learning rate
        gbdt = GBTClassifier(maxIter=250, maxDepth=10, featuresCol='features', labelCol='label',seed=42)


        #CV
        print("Start for cv")
        paramGrid = ParamGridBuilder() \
            .addGrid(gbdt.stepSize, [0.01,0.1]) \
            .build()

        # evaluate using binary classification auc
        evaluator = BinaryClassificationEvaluator(metricName = "areaUnderROC")


        # Create a CrossValidator with the pipeline, ParamGrid, and evaluator
        crossval = CrossValidator(estimator=gbdt,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)

        # Fit the CrossValidator to the data
        cvModel = crossval.fit(trainSet_data)

        # Get the best model
        best_pipeline = cvModel.bestModel

        print("End for cv")

        # Get the best GBTClassifier stage from the pipeline
        best_gbt = best_pipeline.stages[0]

        # Print the best model's hyperparameters
        print("Best Step Size:", best_gbt.getStepSize())

    return 

GBDT_cv()



# %%
gbdtModel.transform(trainSet).select('probability', 'prediction').show(truncate=False)

# %%
to_array = fn.udf(lambda v:v.toArray().tolist(), ArrayType(FloatType()))

# %%
gbdtModel.transform(trainSet).select('probability', 'prediction').withColumn('prediction_score', to_array(fn.col('probability'))[1]).show()

# %%
.withColumn('prediction_score', fn.element_at(fn.col('probability'), 1))

# %%
0.9951352205300913 + 0.004864779469908736
