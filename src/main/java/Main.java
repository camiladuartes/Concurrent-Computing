import java.io.IOException;
import java.util.Scanner;

import knn.AtomicKNN;
import knn.IKNN;
import knn.MutexKNN;
import knn.SerialKNN;
import concurrImprov.CallableAndFutureKNN;
import concurrImprov.ForkAndJoinKNN;
import concurrImprov.ParallelStreamsKNN;
//import concurrImprov.ReactiveProgrammKNN;
import concurrImprov.IKNNimprov;


//=== README
/*
 * >> Run Main:
 * In knnAlgo folder:
 * 1. javac *.java
 * 2. java -Xms2G -Xmx4G Main
 * 
 * >> Run JMH test:
 * 1. Maven Clean, Maven Install
 * 2. java -jar target/benchmarks.jar
 * 
 * >> Run JMeter test:
 * 1. Maven Clean, Maven Install
 * 2. Copy TestesJMeter-0.0.1-SNAPSHOT.jar to apache-jmeter/lib/ext
 * 3. Execute ./bin/jmeter
 * */

public class Main {

	public static void main(String[] args) throws IOException, InterruptedException {
		System.out.println("\n>>> Welcome to KNN Algorithm <<<\n");
		
		String TRAIN_FILENAME = "/home/camiladuartes_/WorkspaceEclipse/knn/diabetes_1_3gb.csv"; // used lines: 40000000 (100%)
		String TEST_FILENAME = "/home/camiladuartes_/WorkspaceEclipse/knn/diabetes_328mb.csv";  // used lines: 200
		int k = 5;
		int NUM_INSTANCES_TRAIN = 50000;
		int NUM_INSTANCES_TEST = 200;
		
		// Get the number of available processors
		Runtime runtime = Runtime.getRuntime();
		int NUM_THREADS = runtime.availableProcessors();
		
		IKNN knn = null;
		IKNNimprov knn_ = null;
		int chooseAlgo = 0;
		Boolean firstImplem = true;
		while(true) {
			System.out.println(">> Choose a KNN Algorithm Implementation:");
			System.out.println("\t1- Serial KNN");
			System.out.println("\t2- Mutex KNN");
			System.out.println("\t3- Atomic KNN");
			System.out.println("\t4- Callable & Future KNN");
			System.out.println("\t5- Fork & Join KNN");
			System.out.println("\t6- Parallel Streams KNN");
			System.out.println("\t7- Reactive Programming KNN");
			
			Scanner s = new Scanner(System.in); 
			chooseAlgo = s.nextInt();
			
			if(chooseAlgo == 1) {
				knn = new SerialKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
				break;
			}
			else if(chooseAlgo == 2) {
				knn = new MutexKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
				break;
			}
			else if(chooseAlgo == 3){
				knn = new AtomicKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
				break;
			}
			if(chooseAlgo == 4) {
				knn_ = new CallableAndFutureKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
				firstImplem = false;
				break;
			}
			else if(chooseAlgo == 5) {
				knn_ = new ForkAndJoinKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
				firstImplem = false;
				break;
			}
			else if(chooseAlgo == 6){
				knn_ = new ParallelStreamsKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
				firstImplem = false;
				break;
			}
//			else if(chooseAlgo == 7){
//				knn_ = new ReactiveProgrammKNN(k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST);
//				firstImplem = false;
//				break;
//			}
		}
		if(firstImplem)
			knn.knnAlgorithm();
		else
			knn_.knnAlgorithm();
	}
}
