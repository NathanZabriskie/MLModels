package learning_models;

import java.util.Random;

import utils.ArrayUtils;
import utils.Matrix;

public class MultiLayerPerceptron extends SupervisedLearner 
{
	private Layer[] layers;
	private Layer[] bestLayers;
	private Random rand;
	private double learningRate;
	
	private Matrix validationSet;
	private Matrix validationLabels;
	private Matrix trainingSet;
	private Matrix trainingLabels;
	
	private double bestAccuracy = -1.0;
	private int sinceImprovement = 0;
	private int epochs = 0;
	
	private final double PERCENT_VALIDATION = 0.3; 
	private final int[] HIDDEN_LAYERS = {10, 20, 10};
	private final int STOP = 500;
	
	private enum nodeType { Output, Hidden };
	
	public MultiLayerPerceptron(Random rand, double learningRate)
	{
		this.rand = rand;
		this.learningRate = learningRate;
		//Add two layers for the input and output layers
		layers = new Layer[HIDDEN_LAYERS.length + 2];
		bestLayers = new Layer[layers.length];
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		features.shuffle(rand, labels);
		initSets(features, labels);
		initLayers(features.cols(), labels.valueCount(0));
		
		double[] result = new double[1];
		double accuracy = 0.;
		epochs = 0;
		do
		{
			epochs++;
			sinceImprovement++;
			trainingSet.shuffle(rand, trainingLabels);
			for(int i = 0; i < trainingSet.rows(); i++)
			{
				predict(trainingSet.row(i), result);
				updateWeights(trainingLabels.row(i)[0]);
			}
			
			int num_correct = 0;
			for(int i = 0; i < validationSet.rows(); i++)
			{
				predict(validationSet.row(i), result);
				if(Math.abs(result[0] - validationLabels.row(i)[0]) < .01)
				{
					num_correct++;
				}
			}
			
			accuracy = num_correct / (double)validationSet.rows();
			if(accuracy > bestAccuracy)
			{
				sinceImprovement = 0;
				bestAccuracy = accuracy;
				for(int i = 0; i < layers.length; i++)
				{
					bestLayers[i] = new Layer(layers[i]);
				}
			}
			else if(sinceImprovement > STOP)
			{
				for(int i = 0; i < layers.length; i++)
				{
					layers[i] = new Layer(bestLayers[i]);
				}
				break;
			}
		} while(true);
		System.out.println("Trained in " + epochs + " epochs with an accuracy of " + accuracy + " on training set.");
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		double[] inputs = features;
		for(int i = 0; i < layers.length; i++)
		{
			inputs = layers[i].calculateOutputs(inputs);
		}
		
		double net = Double.MIN_VALUE;
		for(int i = 0; i < inputs.length; i++)
		{
			if(inputs[i] > net)
			{
				net = inputs[i];
				labels[0] = (double)i;
			}
		}
	}

	private void updateWeights(double target) 
	{
		layers[layers.length - 1].updateWeights((int)target, layers[layers.length - 2], null);
		for(int i = layers.length - 2; i > 0; i--)
		{
			layers[i].updateWeights((int)target, layers[i-1], layers[i+1]);
		}
	}
	
	private void initSets(Matrix features, Matrix labels) 
	{
		int numValidation = (int)(features.rows() * PERCENT_VALIDATION);
		int numTraining = features.rows() - numValidation;
		
		trainingSet = new Matrix(features, 0, 0, numTraining, features.cols());
		trainingLabels = new Matrix(labels, 0, 0, numTraining, labels.cols());
		validationSet = new Matrix(features, numTraining, 0, numValidation - 1, features.cols());
		validationLabels = new Matrix(labels, numTraining, 0, numValidation - 1, labels.cols());
	}

	private void initLayers(int inputCols, int outputCols) 
	{	
		layers[0] = new Layer(inputCols, inputCols, null, nodeType.Hidden);
		
		for(int i = 1; i < layers.length - 1; i++)
		{
			layers[i] = new Layer(HIDDEN_LAYERS[i - 1], layers[i - 1].numNodes(), null, nodeType.Hidden);
		}
		
		layers[layers.length - 1] = new Layer(outputCols, layers[layers.length - 2].numNodes(), null, nodeType.Output);
	}

	private class Layer
	{
		public Node[] nodes;
		
		public Layer(int numNodes, int numInputs, double[] initialWeights, nodeType type)
		{
			this.nodes = new Node[numNodes];
			for(int i = 0; i < this.nodes.length; i++)
			{
				this.nodes[i] = new Node(numInputs + 1, initialWeights, type, i);
			}
		}
		
		public Layer(Layer other)
		{
			nodes = new Node[other.nodes.length];
			for(int i = 0; i < nodes.length; i++)
			{
				nodes[i] = new Node(other.nodes[i]);
			}
		}
		
		public void updateWeights(int target, Layer previousLayer, Layer nextLayer) 
		{
			for(int i = 0; i < nodes.length; i++)
			{
				if(target == i)
				{
					nodes[i].updateWeights(1.0, previousLayer, nextLayer);
				}
				else
				{
					nodes[i].updateWeights(0.0, previousLayer, nextLayer);
				}
			}
		}

		public double[] calculateOutputs(double[] inputs)
		{
			double[] result = new double[nodes.length];
			for(int i = 0; i < nodes.length; i++)
			{
				result[i] = nodes[i].calculateOutput(ArrayUtils.Append(inputs, 1.0)); //Add in the bias term.
			}
			
			return result;
		}
		
		public int numNodes()
		{
			return nodes.length;
		}
		
		private class Node
		{
			public double[] weights;
			public double lastOutput = 0.0;
			public double lastDelta = 0.0;
			public nodeType type;
			public int myIndex;
			
			public Node(int numInputs, double[] initialWeights, nodeType type, int index)
			{
				this.type = type;
				this.myIndex = index;
				
				if(initialWeights == null)
				{
					weights = new double[numInputs];
					for(int i = 0; i < weights.length; i++)
					{
						weights[i] = (rand.nextDouble() * 2) - 1.0;
					}
				} else
				{
					this.weights = initialWeights;
				}
			}
			
			public Node(Node other)
			{
				weights = ArrayUtils.Copy(other.weights);
				this.lastOutput = other.lastOutput;
				this.lastDelta = other.lastDelta;
				this.type = other.type;
				this.myIndex = other.myIndex;
			}
			
			public void updateWeights(double target, Layer previousLayer, Layer nextLayer) 
			{
				Node[] lastNodes = previousLayer.nodes;
				Node[] nextNodes = null;
				double f_prime = lastOutput * (1 - lastOutput);
				double delta = 0.0;
				if(type == nodeType.Output)
				{
					delta = (target - lastOutput) * f_prime;
				}
				else
				{
					nextNodes = nextLayer.nodes;
					for(int i = 0; i < nextNodes.length; i++)
					{
						delta += nextNodes[i].lastDelta * nextNodes[i].weights[myIndex];
					}
					delta *= f_prime;
				}
				
				for(int i = 0; i < weights.length - 1; i++)
				{
					weights[i] += learningRate * lastNodes[i].lastOutput * delta;
				}
				
				weights[weights.length - 1] += learningRate * delta;
				
				lastDelta = delta;
			}

			public double calculateOutput(double[] inputs)
			{
				double net = ArrayUtils.Dot(weights, inputs);
				lastOutput = 1.0 / (1.0 + Math.pow(Math.E, -net));
				return lastOutput;
			}
		}
	}
}
