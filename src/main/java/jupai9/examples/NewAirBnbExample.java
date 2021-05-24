package jupai9.examples;

import java.io.IOException;

        import org.apache.spark.ml.Pipeline;
        import org.apache.spark.ml.PipelineModel;
        import org.apache.spark.ml.PipelineStage;
        import org.apache.spark.ml.feature.VectorAssembler;
        import org.apache.spark.ml.regression.LinearRegression;
        import org.apache.spark.sql.DataFrameReader;
        import org.apache.spark.sql.Dataset;
        import org.apache.spark.sql.Row;
        import org.apache.spark.sql.SparkSession;

public class NewAirBnbExample {

    public static void main(String[] args) throws IOException {
        // Create Spark Session to create connection to Spark
        final SparkSession sparkSession = SparkSession.builder().appName("AirBnb Spark Linear Regression Demo")
                .master("local[5]").getOrCreate();

        // Get DataFrameReader using SparkSession and set header option to true
        // to specify that first row in file contains name of columns
        final DataFrameReader dataFrameReader = sparkSession.read().option("header", true);
        final Dataset<Row> trainingData = dataFrameReader.csv("src/main/resources/listings.csv");

        // Create view and execute query to convert types as, by default, all
        // columns have string types
        trainingData.createOrReplaceTempView("TRAINING_DATA");

        final Dataset<Row> typedTrainingData = sparkSession
                .sql("SELECT cast(id as float) id, cast(bedrooms as float) bedrooms, " +
                            "cast(minimum_nights as float) minimum_nights, " +
                            "cast(number_of_reviews as float) number_of_reviews, " +
                            "cast(price as float) price FROM TRAINING_DATA");
       // final Dataset<Row> notNullTypedTrainingData=typedTrainingData.re
        // Combine multiple input columns to a Vector using Vector Assembler
        // utility
        final VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] { "id", "bedrooms", "minimum_nights", "number_of_reviews", "price"})
                .setOutputCol("features");
        final Dataset<Row> featuresData = vectorAssembler.transform(typedTrainingData.na ().drop ());
        // Print Schema to see column names, types and other metadata
        featuresData.printSchema();

        // Split the data into training and test sets (30% held out for
        // testing).
        Dataset<Row>[] splits = featuresData.randomSplit(new double[] { 0.8, 0.2 },42);
        Dataset<Row> trainingFeaturesData = splits[0];
        Dataset<Row> testFeaturesData = splits[1];


        // Load the model
        PipelineModel model = null;
        try {
            model = PipelineModel.load("src/main/resources/AirbnbPricePrediction");
        } catch(Exception exception) {
        }

        if(model == null) {
            // Train a Linear Regression model.
            final LinearRegression regression = new LinearRegression().setLabelCol("price")
                    .setFeaturesCol("features");

            // Using pipeline gives you benefit of switching regression model without any other changes
            final Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[] { regression });

            // Train model. This also runs the indexers.
            model = pipeline.fit(trainingFeaturesData);
            model.save("src/main/resources/AirbnbPricePrediction");
        }

        // Make predictions.
        final Dataset<Row> predictions = model.transform(testFeaturesData);
        predictions.show();
    }

}