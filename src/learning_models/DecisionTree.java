package learning_models;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import utils.ArrayUtils;
import utils.Matrix;
	
public class DecisionTree extends SupervisedLearner 
{
	public static int numNodes = 0;
	public static int depth = 0;
	
	private Matrix trainingSet;
	private Matrix trainingLabels;
	private Matrix validationSet;
	private Matrix validationLabels;
	
	private DecisionTreeNode rootNode = null;
	
	private double trainAccuracy;
	private double bestAccuracy;
	private double PERCENT_VALIDATION = .3;
	
	private Set<Integer> printedNumbers;
	
	private Random rand;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception 
	{
		rand = new Random();
		numNodes = 0;
		depth = 0;
		printedNumbers = new TreeSet<Integer>();
		initSets(features, labels);
		rootNode = new DecisionTreeNode(features, labels, new TreeSet<Integer>(), 0, null, 0);
		rootNode.split();
		
		trainAccuracy = 0.0;
		bestAccuracy = 0.0;
		double[] res = new double[1];
		for(int i = 0; i < trainingSet.rows(); i++)
		{
			predict(trainingSet.row(i), res);
			if(res[0] == trainingLabels.get(i, 0))
			{
				trainAccuracy += 1.0;
			}
		}
		
		for(int i = 0; i < validationSet.rows(); i++)
		{
			predict(validationSet.row(i), res);
			if(res[0] == validationLabels.get(i, 0))
			{
				bestAccuracy += 1.0;
			}
		}
		trainAccuracy /= trainingSet.rows();
		bestAccuracy /= validationSet.rows();
		System.out.println(String.format("After initial training reached %d nodes and a depth of %d with an accuracy of %f on training.", numNodes, depth, trainAccuracy));
		
		rootNode.prune();
		depth = 0;
		rootNode.updateDepth();
		System.out.println(String.format("After pruning reached %d nodes and a depth of %d with an accuracy of %f on validation.", numNodes, depth, bestAccuracy));
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception 
	{
		labels[0] = rootNode.predict(features);
	}
	
	public static int chooseFeature(Matrix features, Matrix labels, Set<Integer> used_features)
	{
		double min_info = Double.MAX_VALUE;
		int min_i = -1;
		
		for(int i = 0; i < features.cols(); i++)
		{
			if(used_features.contains(i))
			{
				continue;
			}
			//int[][] class_count = new int[features.valueCount(i)][labels.valueCount(0)];
			double feature_info = info(features, labels, i);
			if(feature_info < min_info)
			{
				min_info = feature_info;
				min_i = i;
			}
		}
		
		return min_i;
	}
	
	public double getValidationAccuracy() throws Exception
	{
		double[] res = new double[1];
		double accuracy = 0.0;
		for(int i = 0; i < validationSet.rows(); i++)
		{
			predict(validationSet.row(i), res);
			if(res[0] == validationLabels.get(i, 0))
			{
				accuracy++;
			}
		}
		
		return accuracy / validationSet.rows();
	}
	
	private static double info(Matrix features, Matrix labels, int feature_id) 
	{
		double[][] hist = new double[features.valueCount(feature_id)][labels.valueCount(0)];
		int[][] count = new int[features.valueCount(feature_id)][labels.valueCount(0)];
		
		for(int j = 0; j < features.rows(); j++)
		{
			int index = (int) (features.get(j, feature_id) == Matrix.MISSING ? hist.length - 1 : features.get(j, feature_id));
			hist[index][(int)labels.get(j, 0)]++;
			count[index][(int)labels.get(j, 0)]++;
		}	
		
		double score = 0.0;
		for(int i = 0; i < hist.length; i++)
		{
			int sum = ArrayUtils.Sum(count[i]);
			for(int j = 0; j < (int)labels.valueCount(0); j++)
			{
				if(sum != 0)
				{
					hist[i][j] /= (double) sum;
				}
			}
			
			score += (ArrayUtils.Sum(count[i]) / (double)features.rows()) * entropy(hist[i]);
		}
		
		return score;
	}

	private static double entropy(double[] hist) {
		double res = 0.0;
		for(int i = 0; i < hist.length; i++)
		{
			if(hist[i] == 0)
			{
				continue;
			}
			res += hist[i] * Math.log(hist[i]);
		}
		return -res;
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
	
	private class DecisionTreeNode
	{
		public DecisionTreeNode[] children;
		public DecisionTreeNode parent;
		public Matrix featureSet;
		public Matrix labelSet;
		public Set<Integer> usedFeatures;
		public int splitIndex;	//index of the feature used to split at this node
		public int myIndex; 	//used to keep track of what child this is
		public int[] labelCount;
		public int myGuess = -1;
		public int myDepth;
		
		public DecisionTreeNode(Matrix features, Matrix labels, Set<Integer> usedFeatures, int myIndex, DecisionTreeNode parent, int myDepth) {
			featureSet = features;
			labelSet = labels;
			this.usedFeatures = usedFeatures; 
			this.myIndex = myIndex;
			this.parent = parent;
			
			labelCount = new int[labels.valueCount(0)];
			
			for(int i = 0; i < labels.rows(); i++)
			{
				int index = (int) (labels.get(i, 0) == Matrix.MISSING ? labelCount.length - 1 : labels.get(i, 0));
				labelCount[index]++;
			}
			myGuess = ArrayUtils.MaxIndex(labelCount);
			numNodes++;
			this.myDepth = myDepth;
			DecisionTree.depth = Math.max(DecisionTree.depth, this.myDepth);	
		}
		
		public void updateDepth() 
		{
			DecisionTree.depth = Math.max(myDepth, DecisionTree.depth);
			if(children != null)
			{
				for(int i = 0; i < children.length; i++)
				{
					if(children[i] != null)
					{
						children[i].updateDepth();
					}
				}
			}
		}

		public void prune() throws Exception 
		{	
			DecisionTreeNode temp;
			if(children != null)
			{
				for(int i = 0; i < children.length; i++)
				{
					temp = children[i];
					children[i] = null;
					double acc = getValidationAccuracy();
					if(acc >= bestAccuracy)
					{
						bestAccuracy = acc;
						if(temp != null)
						{
							temp.remove();
						}
					}
					else
					{
						children[i] = temp;
						if(children[i] != null)
						{
							children[i].prune();
						}
					}
				}
			}
		}

		private void remove() 
		{
			if(children != null)
			{
				for(int i = 0; i < children.length; i++)
				{
					if(children[i] != null)
					{
						children[i].remove();
					}
				}
			}
			
			DecisionTree.numNodes--;
		}

		public void split()
		{
			if(ArrayUtils.Max(labelCount) == labelSet.rows())
			{
				myGuess = ArrayUtils.MaxIndex(labelCount);
				return;
			}
			else if(featureSet.rows() == 0 || featureSet.cols() == 0)
			{
				myGuess = ArrayUtils.MaxIndex(parent.labelCount);
				return;
			}
			else if(usedFeatures.size() == featureSet.cols())
			{
				myGuess = ArrayUtils.MaxIndex(parent.labelCount);
				return;
			}
			
			splitIndex = chooseFeature(featureSet, labelSet, usedFeatures);
			/*do
			{
				splitIndex = rand.nextInt(featureSet.cols());
			} while(usedFeatures.contains(splitIndex));*/
			usedFeatures.add(splitIndex);
			/*if(!printedNumbers.contains(splitIndex))
			{
				printedNumbers.add(splitIndex);
				System.out.println(splitIndex);
			}*/
			children = new DecisionTreeNode[featureSet.valueCount(splitIndex)];
			ArrayList<ArrayList<Integer>> childRows = new ArrayList<ArrayList<Integer>>();
			
			for(int i = 0; i < children.length; i++)
			{
				childRows.add(new ArrayList<Integer>());
			}
			
			for(int i = 0; i < featureSet.rows(); i++)
			{
				int index = (int) (featureSet.get(i, splitIndex) == Matrix.MISSING ? childRows.size() - 1 : featureSet.get(i, splitIndex));
				childRows.get(index).add(i);
			}
			
			for(int i = 0; i < children.length; i++)
			{
				children[i] = new DecisionTreeNode(new Matrix(featureSet, childRows.get(i)), new Matrix(labelSet, childRows.get(i)), new TreeSet<Integer>(usedFeatures), i, this, myDepth + 1);
				children[i].split();
			}
		}
		
		public int predict(double[] featureRow)
		{
			if(children == null)
			{
				if(myGuess == -1)
				{
					return ArrayUtils.MaxIndex(labelCount);
				}
				else
				{
					return myGuess;
				}
			}
			double nextNode = featureRow[splitIndex];
			if(nextNode == Matrix.MISSING)
			{
				nextNode = children.length - 1;
			}
			
			if(children[(int)nextNode] == null)
			{
				if(myGuess == -1)
				{
					return ArrayUtils.MaxIndex(labelCount);
				}
				else
				{
					return myGuess;
				}
			}
			else
			{
				return children[(int)nextNode].predict(featureRow);
			}
		}
	}

}
