PRAGMA memory_limit='10GB';
PRAGMA temp_directory='/mnt/nvme1/cmbenello/duckdb_tmp';
PRAGMA max_temp_directory_size='200GB';

-- Load table
CREATE OR REPLACE TABLE lineitem AS
SELECT *
FROM read_csv_auto('/mnt/nvme1/cmbenello/data/sf500/lineitem.csv',
                   header=true);

-- Sort by key columns (l_shipdate, l_extendedprice, l_partkey)
-- and include selected value columns in output
COPY (
    SELECT
        l_shipdate,
        l_extendedprice,
        l_partkey,
        l_orderkey,
        l_linenumber,
        l_commitdate
    FROM lineitem
    ORDER BY
        l_shipdate,
        l_extendedprice,
        l_partkey
)
TO '/mnt/nvme1/cmbenello/data/sf500/lineitem_sorted_duckdb.csv'
WITH (FORMAT CSV, HEADER TRUE);
