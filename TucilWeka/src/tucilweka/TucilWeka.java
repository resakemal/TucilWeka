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
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;

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
        DataSink.write("model.arff", I);
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
    }
    
}
