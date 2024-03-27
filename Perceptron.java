/*
 * Perceptron.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

public class Perceptron extends Classifier {

  // data members

  // methods inherited from Classifier

  // other methods as you see fit

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
