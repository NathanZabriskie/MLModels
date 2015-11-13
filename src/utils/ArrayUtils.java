package utils;

public class ArrayUtils 
{
	public static double Dot(double[] first, double[] second)
	{
		assert first.length == second.length;
		
		double res = 0.0;
		for(int i = 0; i < first.length; i ++)
		{
			res += first[i] * second[i];
		}
		
		return res;
	}
	
	public static double[] Copy(double[] toCopy)
	{
		double[] copy = new double[toCopy.length];
		for(int i = 0; i < toCopy.length; i++)
		{
			copy[i] = toCopy[i];
		}
		
		return copy;
	}
	
	public static double[] Append(double[] array, double val)
	{
		int length = array.length;
		double[] res = new double[length + 1];
		
		for(int i = 0; i < length; i++)
		{
			res[i] = array[i];
		}
		
		res[length] = val;
		
		return res;
	}
	
	public static double AbsDiff(double[] first, double[] second)
	{
		assert first.length == second.length;
		
		double res = 0.0;
		for(int i = 0; i < first.length; i++)
		{
			res += Math.abs(first[i] - second[i]);
		}
		
		return res;
	}
	
	public static double Norm(double[] vector)
	{
		double res = 0.0;
		for(int i = 0; i < vector.length; i++)
		{
			res += Math.pow(vector[i], 2);
		}
		
		return Math.sqrt(res);
	}
	
	public static double Sum(double[] vector)
	{
		double res = 0.0;
		for(int i = 0; i < vector.length; i++)
		{
			res += vector[i];
		}
		
		return res;
	}
	
	public static int Sum(int[] vector)
	{
		int res = 0;
		for(int i = 0; i < vector.length; i++)
		{
			res += vector[i];
		}
		
		return res;
	}
	
	public static double Average(double[] vector)
	{
		double res = Sum(vector);
		return res / vector.length;
	}
	
	public static double Min(double[] vector)
	{
		double res = Double.MAX_VALUE;
		for(int i = 0; i < vector.length; i++)
		{
			res = Math.min(res, vector[i]);
		}
		
		return res;
	}
	
	public static double Max(double[] vector)
	{
		double res = Double.MIN_VALUE;
		for(int i = 0; i < vector.length; i++)
		{
			res = Math.max(res, vector[i]);
		}
		
		return res;
	}
	
	public static int Max(int[] vector)
	{
		int res = Integer.MIN_VALUE;
		for(int i = 0; i < vector.length; i++)
		{
			res = Math.max(res, vector[i]);
		}
		
		return res;
	}
	
	public static int MaxIndex(int[] vector)
	{
		if(vector.length == 1)
		{
			return 0;
		}
		int max = Max(vector);
		for(int i = 0; i < vector.length; i++)
		{
			if(vector[i] == max)
			{
				return i;
			}
		}
		
		return 0;
	}
	
	public static int MinIndex(double[] vector)
	{
		if(vector.length == 1)
		{
			return 0;
		}
		double min = Min(vector);
		for(int i = 0; i < vector.length; i++)
		{
			if(vector[i] == min)
			{
				return i;
			}
		}

		return -1;
	}
	
	public static int MaxIndex(double[] vector)
	{
		if(vector.length == 1)
		{
			return 0;
		}
		double max = Max(vector);
		for(int i = 0; i < vector.length; i++)
		{
			if(vector[i] == max)
			{
				return i;
			}
		}

		return -1;
	}
	
	public static String GetString(double[] vector)
	{
		StringBuilder builder = new StringBuilder("[ ");
		
		for(int i = 0; i < vector.length - 1; i++)
		{
			builder.append(Double.toString(vector[i]) + ", ");	
		}
		builder.append(Double.toString(vector[vector.length-1]) + " ]");
		
		return builder.toString();
	}
}
