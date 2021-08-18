package knn;
import java.util.SortedMap;

public interface IKNN {
	
	public double getEuclidianDistance(double[] a1, double[] a2);
	
	public void knnAlgorithm();
	
	public SortedMap<Double, Double> calculateKNeighbours(double[] currTest);
	
	public void checkClassification(SortedMap<Double, Double> kNeighbours, double[] currTest);
	
}
