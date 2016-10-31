/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tucilweka;
import weka.core.*;
import weka.core.converters.ArffLoader;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author LENOVO
 */
public class TucilWeka {
    Instances I;
    /**
     * @param args the command line arguments
     */
    
    public void loadFile() throws IOException
    {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File("iris.arff"));
        Instances in = loader.getDataSet();
        I = in;
    }
    
    public static void main(String[] args) {
        TucilWeka W = new TucilWeka();
        try {
            W.loadFile();
        } catch (IOException ex) {
            Logger.getLogger(TucilWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println(W.I.toSummaryString());
    }
    
}
