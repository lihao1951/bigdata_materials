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



DataFrame = RDD + Schema,他是懒执行的，不可变的，执行效率比RDD要高

原因是：

1. 拥有定制化内存管理，堆外内存Spark自己管理
2. 优化的执行计划，catalyst

劣势在于：编译期间缺少类型安全检查，在运行期检查



DataSet ：是DataFrame API的扩展

Spark的最新数据抽象，具有类型安全检查 DataSet是强类型的

支持编解码器 样例类的使用

DataFrame = DataSet[Row]

RDD DataFrame DataSet 是可以相互转换的

## Spark-Shell 使用

1. spark是SQL等方法的入口，是SparkSession的实例
2. 通过spark提供的方法读取JSON文件，将JSON文件转换为DataFrame
3. 通过dataframe提供的API来操作数据
4. 通过注册为临时表，用SQL来操作数据

globalTemp 与 普通的TempView不同之处在于 global是全Session可见的，其他session可以使用，在SQL语句中 使用 golbal_temp.表名 来使用



