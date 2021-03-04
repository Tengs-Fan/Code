echo Without JIT
for i in {1..10};
	do
		/usr/bin/time java -Xint BubbleSort.java;
	done

echo With JIT
for i in {1..10};
	do
		/usr/bin/time java BubbleSort.java;
	done
