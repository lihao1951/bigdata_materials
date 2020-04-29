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
   
      - aggregateByKey(zeroValue:U,[partitioner:Partitioner])(seqOp:(U,V)=>U,combOn:(U,U)=>U)在kv对的RDD中，按照Key将value进行分组合并，合并时，将每个value和初始值作为seq函数的参数进行计算，返回的结果作为一个新的kv对，然后再将结果按照key进行合并，最后将每个分组的value传递给combine继续进行计算。
   
      - foldByKey(zeroValue:V)(func:(V,V)=>V):RDD[(K,V)] 
   
        对aggreateByKey的简化操作 其中关于seqop和combOn都是一样的
        
      - sortByKey在一个(K,V)RDD上调用，K必须实现了Ordered接口，返回一个按照Key排序的RDD
   
      - sortBy(func,[ascending],[numTasks])与sortKey类似，但是更加灵活，func是对值进行操作处理，然后对结果排序
   
      - join (K,V) 和(K,W)调用，然后把相同的对应元素堆在一起 (K,(V,W))的RDD，需要注意的是只返回Key在两个RDD中都存在的情况
   
      - cogroup 在(K,V) (K,W)两个RDD上调用，返回一个(K,(iterator[V],iterator[W]))的RDD，不同Key也会返回
   
      - cartesian 笛卡尔积，对两个序列RDD生成对偶的RDD
   
      - pipe 对每一个分区都执行一个perl或者shell脚本，返回输出的RDD
   
        ```bash
        #!/bin/bash
        echo "A"
        while read LINE;do
        	echo ">>>"${LINE}
        done
        ```
   
      - coalesce(numPatitions) 缩减分区数 用于大数据过滤后，提高小数据集的执行效率
   
      - repatition(numPatitions) 根据分区数，重新通过网络混洗所有数据
   
      - repatitionAndSortWithinPartitions(patitioner) 是repatition函数的变种，不同的是此方法在给定的patitioner内部进行排序 效率更高
   
      - glom 对每一个分区形成一个数组 形成的新的RDD类型 RDD[Array[T]]
   
      - mapValues 针对(K,V)这种类型，只对values进行操作
   
      - subtract 计算差的一种函数，rdd1 rdd2，rdd1.subtract(rdd2) 去除两个RDD中相同的元素，保留rdd1中不同的元素
   
      - sample(withReplacement,fraction,seed) 指定随机种子抽样数量为fraction的数据，withReplacement 表示是否是有放回的 返回RDD
   
   3. 行动
   
      - takeSample 是action操作，类似sample，直接返回结果集合
      - reduce(func) 通过func函数聚集RDD中的所有元素
      - collect 以数组形式返回各个元素
      - count 返回RDD中元素个数
      - first 返回第一个元素
      - take 返回前N个元素
      - takeOrdered(N,[ascending]) 返回前N个排序后的结果
      - aggregate(zeroValue)(seqOp,combOp) 类似aggregateByKey 不过是action操作，返回结果类型不需要和原RDD一样
      - fold(zero)(func) aggregate 的简化操作 seqOp 和CombOp 一样
      - saveAsTextFile 文本文件
      - saveAsSequenceFile  序列文件
      - saveAsObjectFile 对象序列化
      - countByKey() 针对(K,V)这种RDD返回(K,Int)这种map
      - foreach(func) 对每一个元素上运行func
   
      当你在RDD中使用到了class的方法或者属性，该class 需要实现java.io.serizlizable接口 
   
   4. 输入输出
   
      1. 文本文件
   
         textFile 输入 
   
         saveAsTextFile 输出
   
      2. JSON文件输入输出
   
         rextFile 自动读取JSON文件
   
   5. 依赖关系
   
      1. 窄依赖：每一个父RDD的Partition最多被子RDD的一个Partition使用
      2. 宽依赖：多个子RDD的Partition会依赖同一个父RDD的Partition，会引起shuffle
      3. 血统： lineage  数据容错
   
   6. DAG图
   
      - 划分stage，根据宽依赖进行划分
      - 每一个action是一个job，每个job里面分好多个stage
   
   7. 持久化
   
      1. 两种方法：persist和cache，cache是persist的特例
      2. persist可以传入不同的存储等级：
         1. NONE
         2. DISK_ONLY
         3. DISK_ONLY_2
         4. MEMORY_ONLY 就是cache
         5. MEMORY_ONLY_2
         6. MEMORY_ONLY_SER 序列化后存储
         7. MEMORY_ONLY_SER_2
         8. MEMORY_AND_DISK
         9. MEMORY_AND_DISK_2
         10. MEMORY_AND_DISK_SER
         11. MEMORY_AND_DISK_SER_2
         12. OFF_HEAP 非堆内存
   
   8. RDD检查点机制 checkpoint
   
      1. checkpoint与cache不一样，checkpoint一般放到HDFS上，实现高容错，是多副本可靠存储，所以依赖链（之前的血统）就不存在了
      2. 首先sc需要设置checkpoint的地址（sc.setCheckpointDir(hdfs://****)），然后RDD直接调用checkpoint就可以(rdd.checkpoint)
   
   9. RDD分区
   
      1. Hash分区和Range分区，多用Hash，取余运算
   
      2. Range ：按照一定的范围映射到某一分区，用到了水塘抽样算法
   
      3. 自定义分区-非常重要
   
         需要继承Partitioner类(入参为num，代表分区个数)，然后重写两个方法
   
         - numPartitions
         - getPartition（key）=> Int 【0 ~ num-1】
   
   10. 累加器和广播变量
   
       1. 累加器
   
          用来对信息进行聚合，所有分片处理时更新共享变量的功能
   
          sc.accumulator(0)
   
          - 首先需要通过SparkContext声明累加器
   
          - 声明过程中需要提供初始值
   
          - 你可以转换操作或者action中直接使用累加器，但不能读取累加器；注意：一般不推荐在转换操作使用累加器，推荐在action中使用
   
          - driver可以读取累加器的结果：累加器.value 
   
          - 自定义累加器：继承AccumulatorV2(In,Out) 重写以下方法
   
            isZero 内部数据结构是否为空
   
            copy 让Spark框架能够调用copy函数产生一个新的相同的累加器实例 分发到每个Executor
   
            reset 重置你的累加器数据结构
   
            add 修改累加器的值
   
            merge 用于合并多个分区的累加器实例
   
            value 返回累加器的值
   
            使用方法：
   
            ```scala
            val accum = new MyAccumulator()
            sc.register(accum,"myAccum")
            accum.add(elem)
            accum.value
            ```
   
       2. 广播变量
   
          高效的分发较大的对象，向所有工作节点发送一个较大的只读值，以供一个或多个节点使用
   
          ```scala
          val boardcast = sc.boardcast(Array(1,2,3))
          boardcast.value
          ```

