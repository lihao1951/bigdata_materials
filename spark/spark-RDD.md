# SparkRDD

1. RDD事Spark的基石，是分布式数据的集合抽象

   1. RDD是不可变的，新的RDD须是上一个RDD转换后得到
   2. RDD是分区的，RDD里面的具体数据是分布在多台数据Executor上的（堆内内存+堆外内存+磁盘）
   3. RDD是弹性的
      - 存储弹性：根据用户配置或当前集群运行情况自动分配到磁盘和内存。用户透明
      - 容错弹性：当RDD数据被删除或者丢失，RDD会通过血统或者检查点机制自动恢复数据。用户透明
      - 计算弹性：分层计算，由 应用->Job->Stage->TaskSet->Task，每一层都有对应的计算的保障和重复机制。保障计算不会由于突发故障而终止
      - 分片弹性：可以根据业务需求或者一些算子来重新调整RDD的数据分布。

2. Spark Core就是在操作RDD

   1. 创建

      1. 集合中创建RDD

         parallize(seq)

         makeRDD(seq)

         makeRDD(seq[(T,seq)]) 可以指定分区 

      2. 外部存储创建RDD

         textFile("path")

      3. 从其他RDD转换

         从另一个RDD转换

   2. 转换

   3. 缓存
   
   4. 行动
   
   5. 输出

