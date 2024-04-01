/*
 * BP.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

import java.lang.reflect.Array;
import java.util.ArrayList;

public class BP extends Classifier {
  private double E_min; //minimum acceptable error for stopping
  private double maxEpochs; //maximum number of epochs
  private double eta; //learning rate
  private double lambda; //parameter for sigmoid transform function
  private int I; // Number of input neurons including bias
  private int J; // Number of hidden neurons including bias
  private int K; // Number of output neurons
  private double[][] V; // Weights from input to hidden layer
  private double[][] W; // Weights from hidden to output layer

  public BP() {
    E_min = 0.1;
    eta = 0.9;
    lambda = 1.0;
    maxEpochs = 50000;
  }

  @Override
  public double classify(ArrayList<Double> example) throws Exception {
    ArrayList<Double> predicted_h = computeHiddenLayerOutput(example);
    double predicted_output = computeOutputLayerOutput(predicted_h);
    
    if (predicted_output > 0) {
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
  public void train(DataSet ds) throws Exception {
    int M = ds.size();
    I = ds.get(0).size(); //number of inputs, inlcuding bias
    K = 1; //output layer
    //Step 1: Initialize weights
    initializeVW();
    for (int q = 0; q < maxEpochs; q++) {
      double E = 0.0;
      for (int m = 0; m < M; m++) {
        ArrayList<Double> x = ds.get(m);
        double y = ds.getLabels().get(m);
        //Step 2a: compute hidden layer output
        ArrayList<Double> h = computeHiddenLayerOutput(x);
        //Step 2b: compute output layer output
        double output = computeOutputLayerOutput(h);
        //Step 3: compute error value
        E += 0.5 * Math.pow(y - output, 2); 
        //Step 4a: compute error signal for output layer
        double delta_output = (y - output) * output * (1 - output);
        //Step 4b: compute error signal for hidden layer
        ArrayList<Double> delta_hidden = new ArrayList<>();
        for (int j = 0; j < J; j++) {
          delta_hidden.add(h.get(j) * (1 - h.get(j)) * W[0][j] * delta_output);
        }
        //Step 5 and 6: update weights
        updateWeights(x, h, delta_output, delta_hidden);
      }
      if (E < E_min) {
        break;
      }
    }
  }

  public void initializeVW() {
    double min = -1.0;
    double max = 1.0;

    V = new double[J][I];
    W = new double[K][J];
    for (int j = 0; j < J; j++) {
      for (int i = 0; i < I; i++) {
        V[j][i] = min + (max - min) * random.nextDouble();
      }
    }
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < J; j++) {
        W[k][j] = min + (max - min) * random.nextDouble();
      }
    }
  } // BP::Initialize

  public double sigmoid(double x) {
    return 1 / (1 + Math.exp(-lambda * x));
  } // BP::sigmoid

  public ArrayList<Double> computeHiddenLayerOutput(ArrayList<Double> x) {
    ArrayList<Double> h = new ArrayList<>();
    for (int j = 0; j < J; j++) {
      double sum = 0.0;
      for (int i = 0; i < I; i++) {
        sum += x.get(i) * V[j][i];
      }
      h.add(sigmoid(sum));
    }
    return h;
  } // BP::computeHiddenLayerOutput

  public double computeOutputLayerOutput(ArrayList<Double> h) {
    double sum = 0.0;
    for (int j = 0; j < J; j++) {
      sum += h.get(j) * W[0][j];
    }
    return sigmoid(sum);
  } // BP::computeOutputLayerOutput

  public void updateWeights(ArrayList<Double> x, ArrayList<Double> h, double delta_output, ArrayList<Double> delta_hidden) {
    //Step 5: Adjust output layer weights
    for (int j = 0; j < J; j++) {
      W[0][j] += eta * delta_output * h.get(j);
    }

    //Step 6: Adjust hidden layer weights
    for (int j = 0; j < J; j++) {
      for (int i = 0; i < I; i++) {
        V[j][i] += eta * delta_hidden.get(j) * x.get(i);
      }
    }
  } // BP::updateWeights

  @Override
  public double holdOut(double p, DataSet ds) throws Exception {
    int trainingSize = (int) (ds.size() * p);
    // Split the dataset
    DataSet trainingSet = new DataSet();
    DataSet testingSet = new DataSet();
    for (int i = 0; i < ds.size(); i++) {
      if (i < trainingSize) {
        trainingSet.add(ds.get(i));
      } 
      else {
        testingSet.add(ds.get(i));
      }
    }
    train(trainingSet);
    return classify(testingSet); //return accuracy
  }

  public void setE_min( double E_min ) {
    this.E_min = E_min;
  } // BP::setE_min

  public void setJ( int J ) {
    this.J = J;
  } // BP::setJ

  public void setMaxEpochs( double maxEpochs ) {
    this.maxEpochs = maxEpochs;
  } // BP::setMaxEpochs

  public void setEta( double eta ) {
    this.eta = eta;
  } // BP::setEta

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("V = ");
    for (double[] row : V) {
      for (double val : row) {
        sb.append(String.format("%6.3f ", val));
      }
      sb.append("\n");
    }

    sb.append("\n W = ");
    for (double[] row : W) {
      for (double val : row) {
        sb.append(String.format("%6.3f ", val));
      }
      sb.append("\n");
    }
    return sb.toString();
  } // BP::toString

  /**
   * main needs to process the command-line arguments -J, -p, -t, and -T.
   * The application logic is specified in the assignment's prompt.
   * See Perceptron.java for one of the use cases.
   */

  public static void main( String args[] ) {
    try {
      double accuracy = 0.0;
      BP bp = new BP();
      DataSet ds = new DataSet();
      ds.load("monks1.tr.dta");
      DataSet ds_test = new DataSet();
      ds_test.load("monks1.te.dta");
      bp.setJ( 3 );
      bp.train( ds );
      accuracy = bp.classify( ds_test );
      System.out.println(bp.toString());
      System.out.println( "accuracy: " + accuracy );
    } // try
    catch ( Exception e ) {
      System.err.println( e.getMessage() );
      e.printStackTrace();
    } // catch
  } // BP::main

} // BP class
