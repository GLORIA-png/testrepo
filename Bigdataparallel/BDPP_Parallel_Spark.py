import os
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark.sql.functions as F 
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.feature import StringIndexer,VectorAssembler

if os.path.exists("HI-Medium_Trans.csv"):
    print("it's here")

class fraudDetector:
    def __init__(self,file_path,feature_cols):
        conf=SparkConf()\
        .setAppName("Laundery")\
        .set("executer.memory","10g")\
        .set("driver.memory","6g")\
        .set("spark.sql.shuffle.partition","200")\
        .set("spark.local.dir","/home/jack/Documents/spark_space")

        self.spark=SparkSession.builder.config(conf=conf).getOrCreate()
        self.feater_column=feature_cols
        self.df=self.read_file(file_path=file_path)
        self.train_df,self.test_df=self.split_classes()
    
    def read_file(self,file_path):
        df=self.spark.read.option("header", "true").csv(file_path).cache()
        df.show(10,truncate=False)
        df.select("Is Laundering").groupBy("Is Laundering").count().show(10,truncate=False)
        df=df.withColumn("Id",F.monotonically_increasing_id())
        for c in self.feater_column:
            df=df.withColumn(c,F.col(c).cast("float"))
        df = df.withColumn("label", F.col("Is Laundering"))
        return df

    def split_classes(self):

        Lt=self.df.filter(F.col("Is Laundering")==1)
        Lf=self.df.filter(F.col("Is Laundering")==0)
        sample_size=17000
        Lt_tr=Lt.sample(False,float(sample_size)/Lt.count(),seed=42)
        Lf_tf=Lf.sample(False,float(sample_size)/Lf.count(),seed=42)
        train_df=Lt_tr.union(Lf_tf)
        test_df=self.df.join(train_df,on="Id",how="left_anti")
        return train_df,test_df
    
    def indexing_string(self,icol,ocol):
        index=StringIndexer(inputCol=icol,outputCol=ocol)
        index_model=index.fit(self.df)
        return index_model
    
    def vector_assembler(self):
        asebler=VectorAssembler(inputCols=self.feater_column,outputCol="features")
        adftr=asebler.transform(self.test_df)
        adfte=asebler.transform(self.train_df)
        return adftr,adfte

        
    def random_forest(self):

        train_df,test_df=self.vector_assembler()
        rf=RandomForestClassifier(labelCol="label",featuresCol="features")

        parm_grid=ParamGridBuilder()\
        .addGrid(rf.maxDepth,[6])\
        .addGrid(rf.impurity,["gini"])\
        .build()

        evaluating=MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
            )

        cv=CrossValidator(
            estimator=rf,
            estimatorParamMaps=parm_grid,
            evaluator=evaluating,
            numFolds=3
        )

        cv_train=cv.fit(train_df)
        best_model=cv_train.bestModel
        print(f"best_model feature importance:{best_model.featureImportances}")
        predict=best_model.transform(test_df)
        accuracy=evaluating.evaluate(predict)
        print(f"accuracy{accuracy}")


def main():

    featers_col=["From Bank","To Bank","Amount Received","Amount Paid","Is Laundering"]
    fr=fraudDetector("HI-Medium_Trans.csv",feature_cols=featers_col)
    fr.random_forest()
    #fr.read_file("HI-Medium_Trans.csv")

if __name__ == '__main__':
    main()

        
