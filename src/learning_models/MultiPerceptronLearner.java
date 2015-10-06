package learning_models;

import java.util.ArrayList;
import java.util.Random;

import utils.Matrix;

public class MultiPerceptronLearner extends SupervisedLearner {

	private PerceptronLearner[] perceptrons;
	private double learningRate;
	private Random rand;
	
	public MultiPerceptronLearner(Random rand, double learningRate) 
	{
		this.rand = rand;
		this.learningRate = learningRate;
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		int numPerceptrons;
		if(labels.valueCount(0) == 2)
		{
			numPerceptrons = 1;
		}
		else
		{
			numPerceptrons = labels.valueCount(0);
		}
		
		perceptrons = new PerceptronLearner[numPerceptrons];
		for(int i = 0; i < perceptrons.length; i++)
		{
			perceptrons[i] = new PerceptronLearner(this.rand, this.learningRate, i);
			perceptrons[i].train(features, labels);
		}
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		double[] nets = new double[perceptrons.length];
		double[] singleResult = new double[1];		
		ArrayList<Integer> activeOutputs = new ArrayList<Integer>();
		
		for(int i=0; i < perceptrons.length; i++)
		{
			nets[i] = perceptrons[i].multiPredict(features, singleResult);
			if(singleResult[0] == 1.0)
			{
				activeOutputs.add(i);
			}
		}
		
		if(perceptrons.length == 1)
		{
			if(activeOutputs.size() == 1)
			{
				labels[0] = (double)perceptrons[0].getTarget();
			}
			else
			{
				labels[0] = 1.0 - (double)perceptrons[0].getTarget();
			}
		}
		else
		{
			if(activeOutputs.size() == 1)
			{
				labels[0] = (double)perceptrons[activeOutputs.get(0)].getTarget();
			}
			else if(activeOutputs.size() == 0)
			{
				double maxNet = Double.MIN_VALUE;
				int maxIndex = 0;
				for(int i = 0; i < perceptrons.length; i++)
				{
					if(nets[i] > maxNet)
					{
						maxNet = nets[i];
						maxIndex = i;
					}
				}
				
				labels[0] = (double)perceptrons[maxIndex].getTarget();
			}
			else
			{
				double maxNet = Double.MIN_VALUE;
				int maxIndex = 0;
				for(int i = 0; i < activeOutputs.size(); i++)
				{
					if(nets[activeOutputs.get(i)] > maxNet)
					{
						maxIndex = activeOutputs.get(i);
					}
				}
				
				labels[0] = (double)perceptrons[maxIndex].getTarget();
			}
		}
	}

}
