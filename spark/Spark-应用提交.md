# 应用提交和调试

## 应用提交

1. 进入Spark的安装目录的bin，调用spark-submit脚本
2. 在脚本后面传入参数
   - --class app的主类
   - --master app运行的模式
     - local
     - local[N]
     - spark://hostname:port
     - mesos://hostnameLport
     - yarn client
     - yran cluster
   - --deploy-mode [可选的] 指定 client/cluster，生产环境一般用cluster，默认是client
   - jar包的位置及应用参数

## 调试

1. 本地调试 master设置为local[N]

2. 本地连接远程集群调试

   把idea客户端作为driver运行，保持和整个Spark集群的连接关系，前提：你的本机和Spark集群是在同一个网段

   1. 把master 设置为Spark集群地址

   2. setJars(自己生成的jar包地址)

   3. setIfMissng("spark.driver.host",本机地址)