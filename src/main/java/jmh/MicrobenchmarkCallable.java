/*
 * Copyright (c) 2014, Oracle America, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 *  * Neither the name of Oracle nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

package jmh;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;

//import concurrImprov.*;
import concurrImprov.IKNNimprov;
import concurrImprov.CallableAndFutureKNN;
import concurrImprov.ReadCSV;

public class MicrobenchmarkCallable{

    static double[][] testDataset, trainDataset;
    static int NUM_THREADS;
    static ExecutorService executorService;
    static List<Callable<String>> rangePredictions;
    
    @State(Scope.Thread)
    public static class BenchmarkState {
        public IKNNimprov knn;
        
        @Setup
        public void setupBenchmark() throws FileNotFoundException, IOException, InterruptedException {
        	ReadCSV readCSV = new ReadCSV();
        	trainDataset = readCSV.readFile("./diabetes_1_3gb.csv", 40000000);
        	testDataset = readCSV.readFile("./diabetes_328mb.csv", 200);
        	
        	Runtime runtime = Runtime.getRuntime();
    		NUM_THREADS = runtime.availableProcessors();    		
    		executorService  = Executors.newFixedThreadPool(NUM_THREADS);
    		rangePredictions = new ArrayList<>(); // List of tasks;

        	// k, TRAIN_FILENAME, TEST_FILENAME, NUM_THREADS, NUM_INSTANCES_TRAIN, NUM_INSTANCES_TEST
        	this.knn = new CallableAndFutureKNN(6324, "./diabetes_1_3gb.csv", "./diabetes_328mb.csv", NUM_THREADS, 40000000, 200);
        	((CallableAndFutureKNN) this.knn).setTrainDataset(trainDataset);
        	((CallableAndFutureKNN) this.knn).setTrainDataset(testDataset);
        }
    }

    @Benchmark
    @Warmup(iterations = 3)
    @Measurement(iterations = 3)
    @BenchmarkMode(Mode.Throughput)
    @Fork(value=1)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testDistance(BenchmarkState state) throws InterruptedException, IOException {
    	for(int i = 0; i < MicrobenchmarkCallable.NUM_THREADS; i++) {		
			// Left, Right: Indexes of the test dataset that the current thread is responsible (size: NUM_INSTANCES_TEST/NUM_THREADS per thread)
			int left = i*(200/MicrobenchmarkCallable.NUM_THREADS);
			int right = (i+1)*(200/MicrobenchmarkCallable.NUM_THREADS);
			
			MicrobenchmarkCallable.rangePredictions.add(new Callable<String> () {
				@Override
				public String call() throws Exception {
					for(int testLine = left; testLine < right; testLine++) {
						// Line i of testDataset
			    		double[] currTest = MicrobenchmarkCallable.testDataset[testLine];
			    		for(int j = 0; j < MicrobenchmarkCallable.trainDataset.length; j++) {
			    			double[] currNeighbour = MicrobenchmarkCallable.trainDataset[j];
			    			state.knn.getEuclidianDistance(currTest, currNeighbour);
			    		}
					}
					return "Predictions Completed";
				}
			});
		}
    	
    	// Invoke all tasks
		List<Future<String>> result = MicrobenchmarkCallable.executorService.invokeAll(MicrobenchmarkCallable.rangePredictions);

		MicrobenchmarkCallable.executorService.shutdown();

		// To stop execution only when all threads ends their execution
		try {
			MicrobenchmarkCallable.executorService.awaitTermination(60, TimeUnit.SECONDS);
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
		
		return;
    }
    
    @Benchmark
    @Warmup(iterations = 3)
    @Measurement(iterations = 3)
    @BenchmarkMode(Mode.Throughput)
    @Fork(value=1)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void testNeighbours(BenchmarkState state) throws InterruptedException, IOException {
    	for(int i = 0; i < MicrobenchmarkCallable.NUM_THREADS; i++) {	
			// Left, Right: Indexes of the test dataset that the current thread is responsible (size: NUM_INSTANCES_TEST/NUM_THREADS per thread)
			int left = i*(200/MicrobenchmarkCallable.NUM_THREADS);
			int right = (i+1)*(200/MicrobenchmarkCallable.NUM_THREADS);
			
			MicrobenchmarkCallable.rangePredictions.add(new Callable<String> () {
				@Override
				public String call() throws Exception {
					for(int testLine = left; testLine < right; testLine++) {
						// Line i of testDataset
			    		double[] currTest = MicrobenchmarkCallable.testDataset[testLine];
			    		state.knn.calculateKNeighbours(currTest);
					}
					return "Predictions Completed";
				}
			});
    	}
    	
    	// Invoke all tasks
		List<Future<String>> result = MicrobenchmarkCallable.executorService.invokeAll(MicrobenchmarkCallable.rangePredictions);

		MicrobenchmarkCallable.executorService.shutdown();

		// To stop execution only when all threads ends their execution
		try {
			MicrobenchmarkCallable.executorService.awaitTermination(60, TimeUnit.SECONDS);
		} catch(InterruptedException e) {
			e.printStackTrace();
		}
    	
    	return;
    }
}