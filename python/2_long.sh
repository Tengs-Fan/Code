#!/bin/bash

echo "Time is  $(date)"

echo "Running $0 with parameter $# and pid $$"


for file in "$@"; do
	grep foobar "$file" > /dev/null 2> /dev/null
	if [[ $? -ne 0 ]]; then
		echo "File $file doesn't have any foobar,adding one"
		echo "# foobar" >> "$file"
	fi
done
