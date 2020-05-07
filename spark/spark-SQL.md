# Spark SQL

## 基础概念

1. Spark 套件中的一个模块 将数据的计算任务转化为SQL的RDD计算，类似于Hive通过SQL的形式应用MapReduce任务
2. 特点：
   1. 与SparkCore无缝集成 
   2. 统一的数据访问方式 标准化的SQL查询
   3. Hive的继承 Spark SQL通过内嵌Hive或者连接已经部署好的Hive实例实现了对Hive语法的集成操作
   4. 标准化的连接方式 SparkSQL 可以通过启动Thrift Server 来支持JDBC ODBC的访问 让自己作为一个BI Server使用

## 数据抽象

RDD(Spark1.0)-> Dataframe(Spark1.3)-> Dataset(Spark1.6)

