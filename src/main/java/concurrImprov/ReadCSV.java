package concurrImprov;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class ReadCSV {

	public double[][] readFile(String DATASET_FILENAME, int NUM_INSTANCES) {
		
		double[][] dataset = new double[NUM_INSTANCES][9];
        try {
        	 FileReader fileReader = new FileReader(DATASET_FILENAME);

        	 try (BufferedReader bufferedReader = new BufferedReader(fileReader)) {
        		 int lcount = 0;
                 int ccount = 0;
        		 String line;
	        	 while((line = bufferedReader.readLine()) != null) {
	        		 if(lcount >= NUM_INSTANCES)
	        			 break;
	        		 ccount = 0;
	        		 String [] arr = line.split(",");
	        		 if(NUM_INSTANCES == 222 || lcount != 0) {
		        		 for(String a : arr) {
		                 	dataset[lcount][ccount] = Double.parseDouble(a);
		                 	ccount++;
		                 }
	        		 }
	                 lcount++;
	        	 }
        	 }
        	 fileReader.close();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
   	 	return dataset;
	}
}
