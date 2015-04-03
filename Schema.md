## Introduction ##

The schema file is used to check the syntax of the SQL queries and to help generate the C codes to execute the queries on GPU. It is a plain text file, each line of which describes a table structure with the following format:

> _tableName_|_columnName_:_type_|_columnName_:_type_|...

_tableName_ specifies the name of the table.

_columnName_:_type_ specifies the name and type of each table column. Currently we support four data types: **Integer**, **Decimal**, **Date** and **Text**. If the type of the column is Text, another filed needs to be added to illustrate the maximum length of the column. This is shown in the following example.

Note that the schema file is case **insensitive**.

## Example ##

SUPPLIER|S\_SUPPKEY:INTEGER|S\_NAME:TEXT:25|S\_ADDRESS:TEXT:25|S\_CITY:TEXT:10|
S\_NATION:TEXT:15|S\_REGION:TEXT:12|S\_PHONE:TEXT:15

The above line describes the data format of the supplier table from Star Schema Benchmark, which has the following columns:

  * S\_SUPPKEY with type Integer.
  * S\_NAME with type Text and length 25.
  * S\_ADDRESS with type Text and length 25.
  * S\_CITY with type Text and length 10.
  * S\_NATION with type Text and length 15.
  * S\_REGION with type Text and length 12.
  * S\_PHONE with type Text and length 15.

In the source code, we have included an example schema file of Star Schema Benchmark located at test/ssb\_test/ssb.schema .