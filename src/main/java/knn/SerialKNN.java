package knn;
import java.io.IOException;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

public class SerialKNN implements IKNN {
	private int k;
	private double[][] testDataset, trainDataset;
	int NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST;
	int accurates;
	
	public SerialKNN(int k, String TRAIN_FILENAME, String TEST_FILENAME, int NUM_INSTANCES_TRAIN, int NUM_INSTANCES_TEST) throws IOException {
		System.out.println("\n>> Serial KNN Algorithm <<\n");
		
		this.k = k;
		this.NUM_INSTANCES_TRAIN = NUM_INSTANCES_TRAIN;
		this.NUM_INSTANCES_TEST = NUM_INSTANCES_TEST;
		this.accurates = 0;
		
		System.out.println("Processing dataset...");
		//====== Comment this part for JMH Test
			ReadCSV readCSV = new ReadCSV();
			trainDataset = readCSV.readFile(TRAIN_FILENAME, NUM_INSTANCES_TRAIN);
			testDataset = readCSV.readFile(TEST_FILENAME, NUM_INSTANCES_TEST);
		//======
	}
	
	public void setTrainDataset(double[][] trainDataset) {
		this.trainDataset = trainDataset;
	}
	
	public void setTestDataset(double[][] testDataset) {
		this.testDataset = testDataset;
	}
	
	/// Compute Euclidian Distance
	public double getEuclidianDistance(double[] a1, double[] a2) {
		double distance = 0.0;
		
		for(int i = 0; i < a1.length-1; i++) {
			distance += Math.pow(a1[i] - a2[i], 2);
		}
		distance = Math.sqrt(distance);
		
		return distance;
	}
	
	/// KNN Algorithm Implementation
	public void knnAlgorithm() {	
		System.out.println("\nExecuting KNN Algorithm...");
		System.out.println("Number of neighbours: " + k + "\n");
		
		// Runs through smaller dataset
		for(int i = 0; i < this.testDataset.length; i++) {
			
			// Line i of testDataset
			double[] currTest = this.testDataset[i];
			
			// To order neighbours by distance (distance, class)
			SortedMap<Double, Double> kNeighbours = calculateKNeighbours(currTest);
			
			// To check if the algorithm classified the current test line correctly
			checkClassification(kNeighbours, currTest);
		}
		
		double accuracy = (accurates/(double)NUM_INSTANCES_TEST)*100;
		System.out.println(">> Final Accuracy: " + accuracy + "% | " + accurates + "/" + NUM_INSTANCES_TEST);
	}
	
	/// Calculates k-Nearest Neighbours
	public SortedMap<Double, Double> calculateKNeighbours(double[] currTest) {	
		double distance = 0.0;
		// To order neighbours by distance (distance, class)
		SortedMap<Double, Double> kNeighbours = new TreeMap<Double, Double>();
		
		// Runs through bigger dataset
		for(int j = 0; j < this.trainDataset.length; j++) {
			
			double[] currNeighbour = this.trainDataset[j];
			
			// Calculate distance from currentTest line (dataTest[i]) with the 222 lines of testDataset
			distance = getEuclidianDistance(currTest, currNeighbour);
			
			// SortedMap doesn't have k neighbours yet, so I add (in order)
			if(kNeighbours.size() < this.k) {
				kNeighbours.put(distance, currNeighbour[currNeighbour.length-1]);
			}
			else {
				// last key is the bigger distance, if it's smaller than it, replaces
				if(distance < kNeighbours.lastKey()) {
					kNeighbours.remove(kNeighbours.lastKey(), kNeighbours.get(kNeighbours.lastKey()));
					kNeighbours.put(distance, currNeighbour[currNeighbour.length-1]);
				}
			}
		}
		return kNeighbours;
	}
	
	public void checkClassification(SortedMap<Double, Double> kNeighbours, double[] currTest) {
		
		double resultSum  = 0.0; // sums how many 0's and 1's
		
		for(Map.Entry<Double, Double> entry : kNeighbours.entrySet()) {
			// getValue gets class (0 or 1)
			resultSum += entry.getValue();
		}
		
		// there are more 0's, so the final result was 0
		if(resultSum < this.k-resultSum) {
			// if the class of current test 'i' is 0, it's correct
			if(currTest[currTest.length-1] == 0)
				accurates++;
		} 
		else {
			if(currTest[currTest.length-1] == 1)
				accurates++;
		}	
	}
}
