package concurrImprov;
import java.util.SortedMap;

public interface IKNNimprov {
	
	public double getEuclidianDistance(double[] a1, double[] a2);
	
	public void knnAlgorithm() throws InterruptedException;
	
	public SortedMap<Double, Double> calculateKNeighbours(double[] currTest);
	
	public void checkClassification(SortedMap<Double, Double> kNeighbours, double[] currTest);
	
}
