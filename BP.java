/*
 * BP.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

public class BP extends Classifier {

  // data members

  // methods inherited from Classifier

  // other methods as you see fit

  public void setJ( int J ) {
    this.J = J;
  } // BP::setJ

  /**
   * main needs to process the command-line arguments -J, -p, -t, and -T.
   * The application logic is specified in the assignment's prompt.
   * See Perceptron.java for one of the use cases.
   */

  public static void main( String args[] ) {
    try {
      // ...
      System.out.println( "accuracy: " + accuracy );
    } // try
    catch ( Exception e ) {
      System.err.println( e.getMessage() );
      e.printStackTrace();
    } // catch
  } // BP::main

} // BP class
