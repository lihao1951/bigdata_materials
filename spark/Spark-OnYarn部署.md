# Yarn部署

## 机器

| 节点  | 功能   |
| ----- | ------ |
| ict50 | Master |
| ict51 | Worker |
| ict52 | Worker |

==注意==在Yarn中其实没有所谓的Master和Worker，当把Spark任务提交到Yarn上去，Yarn会自动进行Driver注册，然后把其他节点当作Worker进行计算

## 解压

spark的目录：/home/bigdata/spark-2.4.5

## 配置

1. 配置conf/spark-env.sh slaves spark-defaults.conf

   ```sh
   export JAVA_HOME=/home/bigdata/jdk1.8
   export SCALA_HOME=/home/bigdata/scala-2.12.8
   export HADOOP_HOME=/home/bigdata/hadoop-2.7.7
   export HADOOP_CONF_DIR=/home/bigdata/hadoop-2.7.7/etc/hadoop
   ---------------------------------------------------------
   ict50
   ict51
   ict52
   ---------------------------------------------------------
   # 利用hdfs管理spark的依赖jar包
   spark.yarn.jars=hdfs://ict50:9000/spark_jars/*
   ```

2. 启动yarn之后 

   首先创建 hdfs 上的/spark_jars，并把spark目录下jars中的jar上传

   然后将该spark-2.4.5 传输到其他节点，那么就可以用脚本启动了

   ```sh
   ./bin/hdfs dfs -mkdir /spark_jars
   ./bin/hdfs dfs -put ../spark-2.4.5/jars/* /spark_jars/
   
   --------------------------------------------------
   在ict50上spark目录下输入
   ./bin/spark-shell --master yarn --deploy-mode client
   ```

3. 查看任务，输入完上述命令，在yarn的web就可以看到相应的任务显示

## 其他

Spark On Yarn 一般用到生产环境（cluster部署模式，client测试）

Standalone的话 一般用于测试 

Local一般用于快速验证