package jmeter;

import java.io.Serializable;

import org.apache.jmeter.config.Arguments;
import org.apache.jmeter.protocol.java.sampler.AbstractJavaSamplerClient;
import org.apache.jmeter.protocol.java.sampler.JavaSamplerContext;
import org.apache.jmeter.samplers.SampleResult;

import knn.*;

public class MacrobenchmarkAtomic extends AbstractJavaSamplerClient implements Serializable {
	
	@Override 
    public Arguments getDefaultParameters() {
		// Argumentos knn
        Arguments defaultParameters = new Arguments();
        defaultParameters.addArgument("k", "5");
        defaultParameters.addArgument("TRAIN_FILENAME", "/home/camiladuartes_/WorkspaceEclipse/TestesJMeter/diabetes_1_3gb.csv");
        defaultParameters.addArgument("TEST_FILENAME", "/home/camiladuartes_/WorkspaceEclipse/TestesJMeter/diabetes_328mb.csv");
        defaultParameters.addArgument("NUM_INSTANCES_TRAIN", "40000000");
        defaultParameters.addArgument("NUM_INSTANCES_TEST", "200");
        defaultParameters.addArgument("NUM_THREADS", "4");
        return defaultParameters; 
    } 
	   
    @Override 
    public SampleResult runTest(JavaSamplerContext javaSamplerContext) {
        String kStr = javaSamplerContext.getParameter("k");
        int k = Integer.parseInt(kStr);
        
        String TRAIN_FILENAME = javaSamplerContext.getParameter("TRAIN_FILENAME");
		String TEST_FILENAME = javaSamplerContext.getParameter("TEST_FILENAME");
		
		String NUM_INSTANCES_TRAIN_STR = javaSamplerContext.getParameter("NUM_INSTANCES_TRAIN");
		int NUM_INSTANCES_TRAIN = Integer.parseInt(NUM_INSTANCES_TRAIN_STR);
		String NUM_INSTANCES_TEST_STR = javaSamplerContext.getParameter("NUM_INSTANCES_TEST");
		int NUM_INSTANCES_TEST = Integer.parseInt(NUM_INSTANCES_TEST_STR);
		
		String NUM_THREADS_STR = javaSamplerContext.getParameter("NUM_THREADS");
        int NUM_THREADS = Integer.parseInt(NUM_THREADS_STR);
        
        SampleResult result = new SampleResult();
        result.sampleStart();

        try {
        	IKNN knn = new AtomicKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
            
        	knn.knnAlgorithm();
            
            result.sampleEnd(); 
            result.setSuccessful(true);
            result.setResponseMessage("Successfully performed action");
            result.setResponseCodeOK();
        } catch (Exception e) {
        	result.sampleEnd();
            result.setSuccessful(false);
            result.setResponseMessage("Exception: " + e);
            result.setResponseCode("500");
        }
              
        return result; 
    }

 
}