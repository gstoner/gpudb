## Query Plan ##
Query execution plan has a large impact on query performance. Currently our tool cannot automatically determine the optimal query execution plan for a given query. Although some optimization techniques like predicate push down have already been adopted to optimize the query plan, it doesn't change the execution sequence of the query operators in the query. For example, considering the following query (q2.1 from Star Schema Benchmark), after the selection operations, the generated query plan by our tool will join lineorder with supplier first, then join the result with part, and finally with ddate. However, the optimal query plan will
join lineorder with part first, then with supplier, and finally with ddate.

```
   select sum(lo_revenue),d_year,p_brand1
   from lineorder,supplier, part,ddate
   where lo_orderdate = d_datekey
         and lo_partkey = p_partkey
         and lo_suppkey = s_suppkey
         and p_category = 'MFGR#12'
         and s_region = 'AMERICA'
         group by d_year,p_brand1
         order by d_year,p_brand1;
```


In this case, when writing the query, users should take into consideration the table statistics such that the generated query plan is the optimal plan.

## Performance Optimization Techniques ##
### Compression ###
Compression is an effective technique to improve query performance on GPUs since it can significantly reduce the amount of data transferred through PCIe bus. Refer to StorageFormat to learn the format of the the column. Users can compress the column using different compression schemes. If it is a new added compression scheme, the GPU driver program and the implementation of each query operator need to be modified to support the new compression scheme.

When running queries on the compressed data, try to choose the data that may maximize the query performance. Users may need to manually modify the driver program (driver.cu) such that the query can access the right data.

### Transfer Overlapping ###
Transfer Overlapping can be adopted to better utilize the PCIe bandwidth and overlap the PCIe transfer with the kernel execution. To utilize this technique, the data need to be stored in the pinned host memory.

Users can edit XML2CODE/config.py and set UVA to 1 so that the generated codes will utilize the transfer overlapping technique.

### Invisible Join ###
Invisible join is an optimization technique for star schema joins proposed in the Sigmod'08 paper "Column-Stores vs. Row-Stores: How Different Are They Really". Users can refer to the paper for details.

To best utilize this technique, dimension tables need to be sorted on certain columns and queries need to be rewritten. The tool currently don't support rewriting the queries at run time automatically. Users need to manually rewrite the queries before utilizing this technique.

## GPU Kernel Configuration ##
The GPU kernel configuration, such as the number of threads in a block, and the number of blocks in the grid, need to be tuned on different GPUs.