#!/usr/bin/env bash

DUCKDB="/mnt/nvme1/cmbenello/duckdb_bin/duckdb"
SQL="sort_lineitem_duckdb.sql"
RUNS=5

total=0

for i in $(seq 1 $RUNS)
do
	    echo "=== Run $i ==="

	        start=$(date +%s.%N)

		    $DUCKDB -c ".read $SQL" > /dev/null

		        end=$(date +%s.%N)

			    runtime=$(echo "$end - $start" | bc)
			        echo "Runtime: $runtime seconds"
				    echo

				        total=$(echo "$total + $runtime" | bc)
				done

				avg=$(echo "$total / $RUNS" | bc -l)
				echo "Average runtime: $avg seconds"
