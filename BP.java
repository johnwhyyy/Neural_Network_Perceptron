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
  private static int J; // Number of hidden neurons including bias
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
    ArrayList<Double> predicted_output = computeOutputLayerOutput(predicted_h);

    // Return the class label with the highest value
    if (predicted_output.get(0) > predicted_output.get(1)) {
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
    int M = ds.size(); //number of examples
    I = ds.get(0).size(); //number of inputs, inlcuding bias
    K = 2; //output layer nubmer of neurons

    //Convert labels to hot-one encoding
    ArrayList<ArrayList<Double>> hotOneLabels = new ArrayList<>();
    for (int i = 0; i < M; i++) {
      if (ds.getLabels().get(i) == 1) {
        hotOneLabels.add(new ArrayList<Double>(){{ add(1.0); add(0.0); }});
      } 
      else {
        hotOneLabels.add(new ArrayList<Double>() {{ add(0.0); add(1.0); }});
      }
    }

    //Step 1: Initialize weights
    initializeVW();
    for (int q = 0; q < maxEpochs; q++) {
      double E = 0.0;
      for (int m = 0; m < M; m++) {
        ArrayList<Double> x = ds.get(m);
        ArrayList<Double> y = hotOneLabels.get(m);
        //Step 2a: compute hidden layer output
        ArrayList<Double> h = computeHiddenLayerOutput(x);
        //Step 2b: compute output layer output
        ArrayList<Double> output = computeOutputLayerOutput(h);
        //Step 3: compute error value
        for (int k = 0; k < K; k++) {
          E += 0.5 * Math.pow(y.get(k) - output.get(k), 2);
        }
        //Step 4a: compute error signal for output layer
        ArrayList<Double> delta_output = new ArrayList<>();
        for (int k = 0; k < K; k++) {
          delta_output.add((y.get(k) - output.get(k)) * output.get(k) * (1 - output.get(k)));
        }
        //Step 4b: compute error signal for hidden layer
        ArrayList<Double> delta_hidden = new ArrayList<>();
        for (int j = 0; j < J; j++) {
          double sum = 0.0;
          for (int k = 0; k < K; k++) {
            sum += delta_output.get(k) * W[k][j];
          }
          delta_hidden.add(h.get(j) * (1 - h.get(j)) * sum);
        }
        
        //Step 5 and 6: update weights
        updateWeights(x, h, delta_output, delta_hidden);
      }
      if (E < E_min) {
        break;
      }
      if (q == maxEpochs) {
        throw new FailedToConvergeException();
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
  } // BP::InitializeVW

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

  public ArrayList<Double> computeOutputLayerOutput(ArrayList<Double> h) {
    ArrayList<Double> output = new ArrayList<>();
    for (int k = 0; k < K; k++) {
      double sum = 0.0;
      for (int j = 0; j < J; j++){
        sum += h.get(j) * W[k][j];
      }
      output.add(sigmoid(sum));
    }
    return output;
  } // BP::computeOutputLayerOutput

  public void updateWeights(ArrayList<Double> x, ArrayList<Double> h, ArrayList<Double> delta_output, ArrayList<Double> delta_hidden) {
    //Step 5: Adjust output layer weights
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < J; j++) {
        W[k][j] += eta * delta_output.get(k) * h.get(j);
      }
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
    sb.append("V =  \n");
    for (double[] row : V) {
      sb.append("[ ");
      for (double val : row) {
        sb.append(String.format("%6.3f ", val));
      }
      sb.append(" ] \n");
    }

    sb.append("\n W = \n");
    for (double[] row : W) {
      sb.append("[ ");
      for (double val : row) {
        sb.append(String.format("%6.3f ", val));
      }
      sb.append(" ] \n");
    }
    return sb.toString();
  } // BP::toString


  /**
   * main needs to process the command-line arguments -J, -p, -t, and -T.
   * The application logic is specified in the assignment's prompt.
   * See Perceptron.java for one of the use cases.
   */

  public static void main( String args[] ) {
    //args = new String[]{"-t","xor.dta", "-J", "4"};
    //args = new String[]{"-t", "monks1.tr.dta", "-T", "monks1.te.dta", "-J", "4"};
    //args = new String[]{"-t", "votes.dta", "-p", "0.8", "-J", "4"};
    //args = new String[]{"-t", "sonar.dta", "-J", "4"};
    //args = new String[]{"-t", "mushroom.dta", "-J", "4", "-p", "0.8"};
    String trainingFileName = null;
    String testingFileName = null;
    Double p = null;
    Double accuracy = null;

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
        case "-J":
          J =(Integer.parseInt(args[++i]));
          break;
      }
    }
    try{
      DataSet traingDataSet = new DataSet();
      DataSet testingDataSet = new DataSet();
      traingDataSet.load(trainingFileName);
      BP bp = new BP();
      //Case 2: have a training file and a testing file
      if (testingFileName != null){
        testingDataSet.load(testingFileName);
        bp.train(testingDataSet);
        accuracy = bp.classify(testingDataSet);
      }
      //Case 3: Provide training file and training proportion p 
      else if (p != null){
        accuracy = bp.holdOut(p,traingDataSet);
      }
      //Case 1: train and classify on trianing data set
      else{
        bp.train(traingDataSet);
        accuracy = bp.classify(traingDataSet);
      }
      System.out.println( bp.toString());
      System.out.println( "Accuracy: " + accuracy );
    }
    catch ( FailedToConvergeException e ) {
        System.out.println( "Failed to converge!" );
      } // catch
    catch ( Exception e ) {
      System.err.println( e.getMessage() );
      e.printStackTrace();
    } // catch
  } // BP::main

} // BP class
