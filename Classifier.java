import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;

public abstract class Classifier {

  // Do not seed the random-number generator because we want random
  // restarts for the back-propagation algorithm.
  protected Random random = new Random();
  protected DecimalFormat df = new DecimalFormat( ".000" );

  public Classifier() { }

  // Classifies the example and returns the predicted class label of -1.0
  // or +1.0.

  abstract public double classify( ArrayList<Double> example ) throws Exception;

  // Classifies the examples of the data set and returns the accuracy.

  abstract public double classify( DataSet ds ) throws Exception;

  // Performs a hold-out evaluation using the data set and p, which is
  // the proportion of examples for the training set.

  abstract public double holdOut( double p, DataSet ds ) throws Exception;

  // Returns a string representation of the classifier.

  abstract public String toString();

  // Constructs a classifier using the data set.

  abstract public void train( DataSet ds ) throws Exception;

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

} // Classifier class
