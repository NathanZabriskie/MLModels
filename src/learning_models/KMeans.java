package learning_models;

import utils.ArrayUtils;
import utils.Matrix;
import java.lang.Math;

public class KMeans extends SupervisedLearner 
{	
	private Matrix feature_matrix;
	private Matrix labels_matrix;
	
	private int k = 3;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		this.feature_matrix = features;
		this.labels_matrix = labels;
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		int[] k_nearest = new int[k];
		double[] nearest_distances = new double[k];
		
		for(int i = 0 ; i < k; i++)
		{
			k_nearest[i] = -1;
			nearest_distances[i] = 1000000 - i * 10;
		}
		
		for(int i = 0; i < feature_matrix.rows(); i++)
		{
			double dist = get_distance(feature_matrix.row(i), features);
			if(dist < ArrayUtils.Max(nearest_distances))
			{
				int idx = ArrayUtils.MaxIndex(nearest_distances);
				k_nearest[idx] = i;
				nearest_distances[idx] = dist;
			}
		}
		
		//This is a continuous result
		if(labels_matrix.valueCount(0) == 0)
		{
			double result = 0.0;
			for(int i = 0; i < k_nearest.length; i++)
			{
				result += labels_matrix.get(k_nearest[i], 0);
			}
			
			result /= (double)k_nearest.length;
			labels[0] = result;
		}
		else
		{
			int[] label_bins = new int[labels_matrix.valueCount(0)];
			
			for(int i = 0; i < k_nearest.length; i++)
			{
				label_bins[(int)labels_matrix.get(k_nearest[i], 0)]++;
			}
			labels[0] = (double) ArrayUtils.MaxIndex(label_bins);
		}	
	}
	
	private double get_distance(double[] first, double[] second)
	{
		double result = 0.0;
		
		for(int i = 0; i < first.length; i++)
		{
			result += Math.pow(first[i] - second[i], 2);
		}
		
		return Math.sqrt(result);
	}
}
