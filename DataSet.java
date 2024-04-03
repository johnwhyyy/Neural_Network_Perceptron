/*
 * DataSet.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class DataSet extends ArrayList<ArrayList<Double> > {
  private ArrayList<Double> labels = new ArrayList<>();

  public DataSet() { 
    super();
  }

  public void load( String filename ) throws Exception{
    BufferedReader br = new BufferedReader(new FileReader(filename));
    String line;
    while ((line = br.readLine()) != null){
      if (line.startsWith("%") || line.trim().isEmpty()) {
        continue;
      }
      String[] tokens = line.split("\\s+");
      ArrayList<Double> example = new ArrayList<>();
      for (int i = 0; i < tokens.length - 1; i++) { // Exclude the last token, which indicates the label
          example.add(Double.parseDouble(tokens[i]));
      }
      double label = Double.parseDouble(tokens[tokens.length - 1]); // add label

      this.add(example);
      labels.add(label);
    }
  }
  public String toString(){
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < this.size(); i++) {
      sb.append("Data: ").append(this.get(i)).append(",label: ").append(labels.get(i)).append("\n");
    }
    return sb.toString();
  }

  public ArrayList<Double> getLabels() {
    return labels;
  }

  public void addLabel(double label) {
    labels.add(label);
  }

  public static void main(String[] args) throws Exception {
    DataSet ds = new DataSet();
    ds.load("monks1.tr.dta");
    System.out.println(ds.toString());
  }

} // DataSet class
