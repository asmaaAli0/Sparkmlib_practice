package jupai9.examples;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.*;
import scala.Tuple2;

import java.math.BigDecimal;
import java.math.MathContext;


public class AirBNBExample {
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);

        // Create Spark Session to create connection to Spark
        final SparkSession sparkSession = SparkSession.builder ().appName ("AirBnb Analysis Demo").master ("local[6]")
                .getOrCreate ();
        // Get DataFrameReader using SparkSession
        final DataFrameReader dataFrameReader = sparkSession.read ();
        // Set header option to true to specify that first row in file contains
        // name of columns
        dataFrameReader.option ("header", "true");
         Dataset<Row> airbnbDF = dataFrameReader.csv ("src/main/resources/listings.csv");
        // Print Schema to see column names, types and other metadata
       // airbnbDF.printSchema ();
        //============================================================================================================
        airbnbDF= airbnbDF.select (  "id","neighbourhood", "room_type", "bedrooms", "minimum_nights",
                "number_of_reviews", "price");
        //final Dataset<Row> airbnbNoNullDF=airbnbDF.na().drop ();
       // airbnbDF.printSchema ();
        System.out.println (airbnbDF.count ());
        System.out.println ("========================================================================");
        //Randomly Split the Dataset to 80 % Train Data and 20 % test Data
        double split[] = {0.8, 0.2};
        Dataset<Row> airbnbDFArray[] = airbnbDF.randomSplit (split, 42);
        //Getting the Train Dataset
        Dataset<Row> airbnbDFTrain = airbnbDFArray[0];
        //ensure that the Train data set field bedrooms  is double and that it does not contain nulls
        airbnbDFTrain = airbnbDFTrain.withColumn ("id", airbnbDFTrain.col ("id").cast ("double"))
                .filter (airbnbDFTrain.col ("id").isNotNull ());
        airbnbDFTrain = airbnbDFTrain.withColumn ("minimum_nights", airbnbDFTrain.col ("minimum_nights").cast ("double"))
                .filter (airbnbDFTrain.col ("minimum_nights").isNotNull ());
        airbnbDFTrain = airbnbDFTrain.withColumn ("number_of_reviews", airbnbDFTrain.col ("number_of_reviews").cast ("double"))
                .filter (airbnbDFTrain.col ("number_of_reviews").isNotNull ());
        airbnbDFTrain = airbnbDFTrain.withColumn ("bedrooms", airbnbDFTrain.col ("bedrooms").cast ("double"))
                .filter (airbnbDFTrain.col ("bedrooms").isNotNull ());
        airbnbDFTrain = airbnbDFTrain.withColumn ("price", airbnbDFTrain.col ("price").cast ("double"))
                .filter (airbnbDFTrain.col ("price").isNotNull ());
        airbnbDFTrain.printSchema ();
        //============================================================================================================
        //Getting the Test Dataset
        Dataset<Row> airbnbDFTest = airbnbDFArray[1];
        //ensure that the Test data set field bedrooms  is double and that it does not contain nulls
        airbnbDFTest = airbnbDFTest.withColumn ("id", airbnbDFTest.col ("id").cast ("double"))
                .filter (airbnbDFTest.col ("id").isNotNull ());
        airbnbDFTest = airbnbDFTest.withColumn ("minimum_nights", airbnbDFTest.col ("minimum_nights").cast ("double"))
                .filter (airbnbDFTest.col ("minimum_nights").isNotNull ());
        airbnbDFTest = airbnbDFTest.withColumn ("number_of_reviews", airbnbDFTest.col ("number_of_reviews").cast ("double"))
                .filter (airbnbDFTest.col ("number_of_reviews").isNotNull ());
        airbnbDFTest = airbnbDFTest.withColumn ("bedrooms", airbnbDFTest.col ("bedrooms").cast ("double"))
                .filter (airbnbDFTest.col ("bedrooms").isNotNull ());
        airbnbDFTest = airbnbDFTest.withColumn ("price", airbnbDFTest.col ("price").cast ("double"))
                .filter (airbnbDFTest.col ("price").isNotNull ());
        airbnbDFTest.printSchema ();
        //============================================================================================================
        System.out.println ("Training Data Set Size is " + airbnbDFTrain.count ());
        System.out.println ("Test Data Set Size is " + airbnbDFTest.count ());
        String inputColumns[] = {"bedrooms"};
        //============================================================================================================
        //Create the Vector Assembler That will contain the feature columns
        VectorAssembler vectorAssembler = new VectorAssembler ();
        vectorAssembler.setInputCols (inputColumns);
        vectorAssembler.setOutputCol ("features");

        //============================================================================================================
        //Transform the Train Dataset using vectorAssembler.transform
        Dataset<Row> airbnbDFTrainTransform = vectorAssembler.transform (airbnbDFTrain.na ().drop ());
        airbnbDFTrainTransform.printSchema ();
        airbnbDFTrainTransform.select ("bedrooms", "features", "price").show (10);
        //Create a LinearRegression Estimator and set the feature column and the label column
        //call linearRegression.fit (airbnbDFTrain) to return a LinearRegressionModel Object
        LinearRegression linearRegression=new LinearRegression ();
        linearRegression.setFeaturesCol ("features");
        linearRegression.setLabelCol ("price");

        LinearRegressionModel linearRegressionModel=linearRegression.fit (airbnbDFTrainTransform);

        double coefficient=Math.round(linearRegressionModel.coefficients ().toArray ()[0]);
        double intercept=Math.round(linearRegressionModel.intercept ());
        System.out.println("The formula for the linear regression line is price = "+coefficient+"*bedrooms + "+intercept);
        airbnbDFTest=vectorAssembler.transform (airbnbDFTest.na().drop ());
        final Dataset<Row> predictions = linearRegressionModel.transform(airbnbDFTest);

        System.out.println("###############################################################");

        predictions.show();
        ///######################################## trying to get accuracy ##################################################///
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("price")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

    }
}
