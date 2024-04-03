/*
 * Perceptron.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Perceptron extends Classifier {
  
  protected double eta = 0.9;
  protected int maxEpochs = 50000;
  protected double threshold = 0.0;
  protected ArrayList<Double> weights = new ArrayList<>();

  // methods inherited from Classifier

  @Override
  public double classify( ArrayList<Double> example ) throws Exception{
    double predicted_label = 0.0;
    for (int i = 0 ; i < example.size(); i++) {
      predicted_label += example.get(i) * weights.get(i);
    }
    if (predicted_label > threshold) {
      return 1.0;
    }
    else {
      return -1.0;
    }
  }

  @Override
  public double classify(DataSet ds) throws Exception {
    ArrayList<Double> predictedLabels = new ArrayList<>();
    for (int i = 0; i < ds.size(); i++) {
      double label = classify(ds.get(i));
      predictedLabels.add(label);
    }
    return computeAccuracy(ds.getLabels(), predictedLabels);
  }

  @Override
  public double holdOut(double p, DataSet ds) throws Exception {
    // Calculate the size of the training set
    int trainSize = (int) (ds.size() * p);
          
    // Split the dataset
    DataSet trainingSet = new DataSet();
    DataSet testingSet = new DataSet();
    for (int i = 0; i < ds.size(); i++) {
      if (i < trainSize) {
        trainingSet.add(ds.get(i));
        trainingSet.addLabel(ds.getLabels().get(i));
      } 
      else {
        testingSet.add(ds.get(i));
        testingSet.addLabel(ds.getLabels().get(i));
      }
    }
    train(trainingSet);
    return classify(testingSet); //return accuracy
  }

  @Override
  public void train(DataSet ds) throws Exception {
    int epoch = 0;
    weights = new ArrayList<Double>(Collections.nCopies(ds.get(0).size(), 0.0)); // Initialize weights
    boolean converged = false;
    
    while (!converged && epoch < maxEpochs){
      converged = true;
      for (int i = 0; i < ds.size(); i++) {
        // Compute the dot product for all rows
        double dotProduct = 0.0;
        for (int j = 0; j < ds.get(i).size(); j++) {
          dotProduct += ds.get(i).get(j) * weights.get(j);
        }
        double y = dotProduct * ds.getLabels().get(i);
        //adjust the weights
        if (y <= 0) {
          for (int j = 0; j < ds.get(i).size(); j++) {
            weights.set(j, weights.get(j) + eta * ds.get(i).get(j) * ds.getLabels().get(i));
          }
          converged = false;
        }
      }
      epoch++;
    }
    if (epoch == maxEpochs) {
      throw new FailedToConvergeException();
    }
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < weights.size(); i++) {
      sb.append("Weight ").append(i).append(": ").append(weights.get(i)).append("\n");
    }
    return sb.toString();
  }

  public void setWeights(ArrayList<Double> weights) {
    if (weights.size() != this.weights.size()) {
      throw new IllegalArgumentException("The size of the weights must be the same as the size of the weights of the model.");
    }
    this.weights = weights;
  }

  public ArrayList<Double> getWeights() {
    return weights;
  }

  /**
   * main needs to process the command-line arguments -p, -t, and -T.
   * The application logic is specified in the assignment's prompt.
   */

  public static void main(String[] args) throws Exception {
    for (int i = 0; i < args.length; i++) {
      System.out.println(args[i]);
    }
    //args = new String[]{"-t", "mushroom.dta"};
    //args = new String[]{"-t", "monks1.tr.dta", "-T", "monks1.te.dta"};
    //args = new String[]{"-t", "mushroom.dta", "-p", "0.9"};
    //args = new String[]{"-t", "xor.dta"};
    String trainingFileName = null;
    String testingFileName = null;
    Double p = null;
    Double accuracy =null;

    //parse command line arguments
    for (int i = 0; i < args.length; i++){
      switch (args[i]){
        case "-t":
          trainingFileName = args[++i];
          break;
        case "-T":
          testingFileName = args[++i];
          break;
        case "-p":
          p = Double.parseDouble(args[++i]);
          break;
      }
    }
    try{
      DataSet traingDataSet = new DataSet();
      DataSet testingDataSet = new DataSet();
      traingDataSet.load(trainingFileName);
      Perceptron perceptron = new Perceptron();
      //Case 2: have a training file and a testing file
      if (testingFileName != null){
        testingDataSet.load(testingFileName);
        perceptron.train(testingDataSet);
        accuracy = perceptron.classify(testingDataSet);
      }
      //Case 3: Provide training file and training proportion p 
      else if (p != null){
        accuracy = perceptron.holdOut(p,traingDataSet);
      }
      //Case 1: train and classify on the same data set
      else{
        testingDataSet.load(trainingFileName);
        perceptron.train(traingDataSet);
        accuracy = perceptron.classify(testingDataSet);
      }
      System.out.println( perceptron.toString() );
      System.out.println( "Accuracy: " + accuracy );
    }
    catch ( FailedToConvergeException e ) {
        System.out.println( "Failed to converge!" );
      } // catch
    catch ( Exception e ) {
      System.err.println( e.getMessage() );
      e.printStackTrace();
    } // catch

  }

} // Perceptron class
