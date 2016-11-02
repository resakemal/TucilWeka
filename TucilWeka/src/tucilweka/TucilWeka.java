/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;
//import weka.core.converters.ArffLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
//import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;

/**
 *
 * @author LENOVO
 */
public class TucilWeka {
    Instances I, I2;
    Evaluation E;
    J48 C;
    /**
     * @param args the command line arguments
     */
    
    public void loadFile() throws IOException, Exception
    {
        //ArffLoader loader = new ArffLoader();
        //loader.setSource(new File("iris.arff"));
        //Instances in = loader.getDataSet();
        I = DataSource.read("iris.arff");
        I.setClassIndex(I.numAttributes() - 1);
    }
    
    public void filter() throws Exception
    {
        NumericToNominal D = new NumericToNominal();
        D.setInputFormat(I);
        I2 = Filter.useFilter(I, D);
    }
    
    public void evaluate() throws Exception
    {
        Evaluation eval = new Evaluation(I);
        C = new J48();
        eval.crossValidateModel(C, I, 10, new Random(1));
        E = eval;
    }
    
    public void buildClassifier() throws Exception
    {
        C = new J48();
        C.buildClassifier(I);
    }
    
    public void saveModel() throws Exception
    {
        weka.core.SerializationHelper.write("tucil.model", C);
    }
    
    public void openModel() throws Exception
    {
         C = (J48) (Classifier) weka.core.SerializationHelper.read("tucil.model");
    }
    
    public void readNewInstance()
    {
        /*
        @ATTRIBUTE sepallength	REAL
        @ATTRIBUTE sepalwidth 	REAL
        @ATTRIBUTE petallength 	REAL
        @ATTRIBUTE petalwidth	REAL
        @ATTRIBUTE class 	{Iris-setosa,Iris-versicolor,Iris-virginica}
        5.5,2.4,3.7,1.0,Iris-versicolor
        */
        Instance inst = new DenseInstance(I.firstInstance());
        // Set instance's values for the attributes "length", "weight", and "position"
        inst.setDataset(I);
        Scanner s1 = new Scanner(System.in);
        String Input;
        Input = s1.next();
        System.out.println(Input);
        s1.close();
        Scanner s = new Scanner(Input).useDelimiter(",");
        while(s.hasNext()){
            inst.setValue(0, s.nextFloat());
            inst.setValue(1, s.nextFloat());
            inst.setValue(2, s.nextFloat());
            inst.setValue(3, s.nextFloat());
            inst.setValue(4, s.next());
        }
        s.close();
        // Set instance's dataset to be the dataset "race"
        I.add(inst);
        // Print the instance
        System.out.println("The instance: " + inst);
    }
    
    public static void main(String[] args) throws Exception {
        TucilWeka W = new TucilWeka();
        try {
            W.loadFile();
        } catch (IOException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println(W.I.toSummaryString());
        W.filter();
        System.out.println(W.I2.toSummaryString());
        W.evaluate();
        System.out.println(W.E.toSummaryString());
        W.buildClassifier();
        W.saveModel();
        System.out.println("\nDataset:\n");
        System.out.println(W.I);
        System.out.println("\nSebelum\n");
        W.readNewInstance();
        System.out.println("\nSesudah\n");
        System.out.println("\nDataset:\n");
        System.out.println(W.I);
    }
    
}
