/*
 * Perceptron.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

import java.util.ArrayList;

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
    return predicted_label;
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
