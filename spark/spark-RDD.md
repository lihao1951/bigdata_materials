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

      转换操作均为懒执行
   
      主要面对两种RDD类型：1. 数值 2.键值对
   
      - map(func) 将函数应用于其中每一个元素，不会把组之间打平
   
      - filter(func) 过滤 将函数返回为true的元素返回
   
      - flatMap(func) 类似于map，返回一个序列，将组之间数据压平
   
      - mapPartitions(func) 类似于map，但是独立在RDD的每一个分区上进行，因此在类型为T的RDD上运行时，func函数的类型必须是Iterator[T]=>Iterator[U]
   
      - mapPartitionsWithIndex(func) 类似于mapPartiitons，每个分区运行一次，输入参数为(Int,Iterator[T])=>Iterator[U]
   
      - union(otherDataset) 对源RDD和RDD求并集，返回新的RDD
   
      - intersection() 求交集，返回新的RDD
   
      - distinct([numTasks]) 对源RDD去重，返回一个新的RDD，默认情况下有8个并行任务执行操作，但可以传入一个numTasks来改变它
   
      - partitionBy 根据分区器对RDD进行分区，若原有的partitonRDD和现在的一致，则不进行分区
   
        可以传入一个分区器比如org.apache.spark.HashPartitoner(分区数:Int)
   
      - reduceByKey 在一个(K,V)上进行调用，返回一个(K,V)的RDD，将相同的值放到一起，先在自己的分区进行聚合，然后再shuffle聚合
   
      - groupByKey 对每一个Key操作，但生成的是每个key有一个相关的sequence
   
      - combineByKey (createCombiner: V => C,mergeValue: (C, V) => C,mergeCombiners: (C, C) => C)当遇到的key没出现，执行createCombiner，如果遇到了，执行mergeValue，最后把不同分区按照mergerCombiner合并
   
      - aggregateByKey(zeroValue:U,[partitioner:Partitioner])(seqOp:(U,V)=>U,combOn:(U,U)=>U)在kv对的RDD中，按照Key将value进行分组合并，合并时，将每个value和初始值作为seq函数的参数进行计算，返回的结果作为一个新的kv对，然后再将结果按照key进行合并，最后将每个分组的value传递给combine继续宁计算。
   
   3. 缓存
   
   4. 行动
   
      - sample(withReplacement,fraction,seed) 指定随机种子抽样数量为fraction的数据，withReplacement 表示是否是有放回的
      - takeSample 是action操作，类似sample，直接返回结果集合
   
   5. 输出

