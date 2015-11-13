package utils;

import java.util.Map;
import java.util.TreeMap;

public class MinMaxMap 
{
	public Map<Integer, MinMax> mmMap;
	
	public MinMaxMap()
	{
		mmMap = new TreeMap<Integer, MinMax>();
	}
}
