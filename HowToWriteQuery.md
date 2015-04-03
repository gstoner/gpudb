Currently we only support queries that have similar formats as Star Schema Benchmark queries. We have included Star Schema Benchmark queries in the directory test/ssb\_test .

When the query has join operations, the first table in the query should be the fact table, and the rest tables should be the dimension tables. For example,

> select A.col, B.col from A, B where A.key = B.key;

In the above query, A should be the fact table and B should be the dimension table. If the dimension tables come before fact table, the generated code may not work correctly.