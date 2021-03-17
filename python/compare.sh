echo Without JIT
for i in {1..10};
	do
		/usr/bin/time python BubbleSort.py;
	done

echo With JIT
for i in {1..10};
	do
		/usr/bin/time python BubbleSort_JIT.py;
	done
