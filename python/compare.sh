echo Without JIT
for i in {1..10};
	do
		/usr/bin/time Python3 BubbleSort.py;
	done

echo With JIT
for i in {1..10};
	do
		/usr/bin/time Python3 BubbleSort_JIT.py;
	done
