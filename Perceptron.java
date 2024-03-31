/*
 * Perceptron.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Perceptron extends Classifier {
  
  protected double learningRate = 0.9;
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
    // Add the contribution from the bias, which is the last weight[n] * 1
    predicted_label += weights.get(weights.size() - 1);
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

  public double computeAccuracy(ArrayList<Double> actualLabels, ArrayList<Double> predictedLabels) {
    if (actualLabels.size() != predictedLabels.size()) {
        throw new IllegalArgumentException("The size of actual labels and predicted labels must be the same.");
    }

    double a = 0; // True Positives
    double b = 0; // True Negatives
    double c = 0; // False Positives
    double d = 0; // False Negatives

    for (int i = 0; i < actualLabels.size(); i++) {
      if (actualLabels.get(i) == 1 && predictedLabels.get(i) == 1) {
          a++; // Correctly predicted positive
      } else if (actualLabels.get(i) == -1 && predictedLabels.get(i) == -1) {
          b++; // Correctly predicted negative
      } else if (actualLabels.get(i) == -1 && predictedLabels.get(i) == 1) {
          c++; // Incorrectly predicted positive
      } else if (actualLabels.get(i) == 1 && predictedLabels.get(i) == -1) {
          d++; // Incorrectly predicted negative
      }
    }
  return (a + b) / (a + b + c + d);
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
    } 
    else {
      testingSet.add(ds.get(i));
    }
  }
  Perceptron trainingModel = new Perceptron();
  trainingModel.train(trainingSet);
  return trainingModel.classify(testingSet); //return accuracy
}

@Override
public void train(DataSet ds) throws Exception {
  weights = new ArrayList<>(Collections.nCopies(ds.get(0).size() + 1, 0.0));// Initialize weights with an additional bias weight
  boolean converged = false;
  int epoch = 0;
  while (!converged && epoch < maxEpochs){
    converged = true;
    for (int i = 0; i < ds.size(); i++) {
     // Compute the dot product for all rows, bias is added at the end
     double dotProduct = 0.0;
     for (int j = 0; j < ds.get(i).size(); j++) {
       dotProduct += ds.get(i).get(j) * weights.get(j);
     }
     // Adding the bias (which is the last weight, interacting with the implicit bias input of 1)
     dotProduct += weights.get(weights.size() - 1);

    if (dotProduct <= 0) {
      for (int j = 0; j < ds.get(i).size(); j++) {
          weights.set(j, weights.get(j) + learningRate * ds.get(i).get(j) * ds.getLabels().get(i));
        }
        converged = false;
      }
    }
    epoch++;
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

  public static void main( String args[] ) {
    try {
      // One use case: train and test on the same examples.
      DataSet ds = new DataSet();
      ds.load("mushroom.dta");
      Perceptron perceptron = new Perceptron();
      perceptron.train(ds);
      double accuracy = perceptron.classify(ds);
      System.out.println( "Accuracy: " + accuracy );
    } // try
    catch ( FailedToConvergeException e ) {
      System.out.println( "Failed to converge!" );
    } // catch
    catch ( Exception e ) {
      System.err.println( e.getMessage() );
      e.printStackTrace();
    } // catch
  } // Perceptron::main

} // Perceptron class
