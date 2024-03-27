/*
 * DataSet.java
 * Copyright (c) 2024 Marcus A. Maloof.  All Rights Reserved.  See LICENSE.
 */

public class DataSet extends ArrayList<ArrayList<Double> > {

  public DataSet() { }
  public void load( String filename ) throws Exception;
  public String toString();

} // DataSet class
