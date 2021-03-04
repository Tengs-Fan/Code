import java.util.Random;
// Java program for implementation of Bubble Sort 
class BubbleSort 
{ 
	void bubbleSort(int arr[]) 
	{ 
		int n = arr.length; 
		for (int i = 0; i < n-1; i++) 
			for (int j = 0; j < n-i-1; j++) 
				if (arr[j] > arr[j+1]) 
				{ 
					// swap arr[j+1] and arr[j] 
					int temp = arr[j]; 
					arr[j] = arr[j+1]; 
					arr[j+1] = temp; 
				} 
	} 

	/* Prints the array */
	void printArray(int arr[]) 
	{ 
		int n = arr.length; 
		for (int i=0; i<n; ++i) 
			System.out.print(arr[i] + " "); 
		System.out.println(); 
	} 

	// Driver method to test above 
	public static void main(String args[]) 
	{ 
		BubbleSort ob = new BubbleSort(); 
		Random rd = new Random(); // creating Random object
		int[] arr = new int [10000];
	    for (int i = 0; i < arr.length; i++) {
			arr[i] = rd.nextInt(); // storing random integers in an array
			//System.out.println(arr[i]); // printing each array element
		}
		ob.bubbleSort(arr); 
		//System.out.println("Sorted array"); 
		//ob.printArray(arr); 
	} 
} 
/* This code is contributed by Rajat Mishra */

