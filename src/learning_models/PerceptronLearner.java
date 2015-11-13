package learning_models;

import java.io.PrintWriter;
import java.util.Random;

import utils.ArrayUtils;
import utils.Matrix;

public class PerceptronLearner extends SupervisedLearner 
{

	private double[] weights;
	private double learningRate;
	private Random rand;
	private int epochs = 0;
	private int targetIndex;
		
	public PerceptronLearner(Random rand, double learningRate, int targetIndex) 
	{
		this.rand = rand;
		this.learningRate = learningRate;
		this.targetIndex = targetIndex;
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		StringBuilder builder = new StringBuilder();
		//Weights have one extra column for the bias input.
		weights = new double[features.cols() + 1];
		double totalMisclassification = 0.0;
		for(int i = 0; i < weights.length; i++)
		{
			weights[i] = 0.0;
		}

		double predictedValue;
		double res = 0;
		int numCorrect = 0;
		double[] withBias;
		double accuracy = 0.0;
		double bestAccuracy = 0.0;
		int sinceImprovement = 0;
		do
		{
			epochs++;
			features.shuffle(rand,labels);
			for(int i = 0; i < features.rows(); i++)
			{
				withBias = ArrayUtils.Append(features.row(i),1.0);
				res = ArrayUtils.Dot(withBias, weights);

				predictedValue = getPrediction(res);
				//System.out.println(labels.attrValue(0, 1));
				if((((int)labels.get(i, 0) == targetIndex) && predictedValue == 1.0) || (((int)labels.get(i, 0) != targetIndex) && predictedValue == 0.0))
				{
					numCorrect++;
				}
				else
				{
					for(int j = 0; j < withBias.length; j++)
					{
						weights[j] += learningRate * ((1-predictedValue) - predictedValue) * withBias[j];
					}
				}
			}
			
			accuracy = numCorrect / (double)features.rows();
			totalMisclassification += 1-accuracy;
			builder.append(epochs);
			builder.append(",");
			builder.append(totalMisclassification / epochs * 100);
			builder.append("\n");
			if(accuracy > bestAccuracy)
			{
				sinceImprovement = 0;
				bestAccuracy = accuracy;
			}
			else
			{
				sinceImprovement++;
			}
			
			if(numCorrect == features.rows() || sinceImprovement > 300)
			{
				System.out.println(ArrayUtils.GetString(weights));
				accuracy = bestAccuracy;
				break;
			}
			numCorrect = 0;
		} while(true);
		
		System.out.println("Trained in " + Integer.toString(epochs) + " epochs and acheived accuracy of " + accuracy * 100 + "%.");
		PrintWriter out = new PrintWriter("resource\\out.csv");//BufferedWriter out = new BufferedWriter(new FileWriter("resource\\out.csv"));
		out.print(builder.toString());
		out.close();
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		double res = 0.0;
		double[] withBias = ArrayUtils.Append(features, 1.0);
		res = ArrayUtils.Dot(withBias, weights);
		
		labels[0] = getPrediction(res);
	}

	public double multiPredict(double[] features, double[] labels) throws Exception 
	{
		double res = 0.0;
		double[] withBias = ArrayUtils.Append(features, 1.0);
		res = ArrayUtils.Dot(withBias, weights);
		
		labels[0] = getPrediction(res);
		return res;
	}
	
	public int getTarget()
	{
		return targetIndex;
	}
	
	private double getPrediction(double sum)
	{
		if (sum > 0)
		{
			return 1.0;
		}
		else
		{
			return 0.0;
		}
	}
}
