# Hadoop-普通集群部署

## 准备节点

将以下加入/etc/hosts

- 10.20.8.50 ict50
- 10.20.8.51 ict51
- 10.20.8.52 ict52

其中对于每个节点规划的功能如下

| 节点  | Hadoop相关功能            | Yarn相关功能    |
| ----- | :------------------------ | --------------- |
| ict50 | NameNode                  | ResourceManager |
| ict51 | DataNode,jobhistory       | NodeManager     |
| ict52 | DataNode,SecondryNameNode | NodeManager     |

## 创建用户并进行免密登录

每个机器创建biddata用户 以ict50作为主服务器 配置其对其他节点的免密登录

```sh
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa # 每个服务器都运行，回车 自动成功，可在~/.ssh下查看
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys # 在ict50下进行
ssh localhost # 在ict50下实验，如果需要输入密码，ssh免密没有成功，需要排查问题；否则成功
# 成功后将密码发给其他节点用户下
scp ~/.ssh/authorized_keys bigdata@ict51:~/.ssh/
scp ~/.ssh/authorized_keys bigdata@ict52:~/.ssh/
# 这样实验从ict50 免密到其他节点
ssh ict51
```

最后每个服务器关闭防火墙

```sh
systemctl stop firewalld
systemctl disable firewalld
```

## 配置JDK和Scala

JDK版本：1.8 （不要openjdk）

Scala版本：2.12.8

上传压缩包到bigdata用户下，解压

在~/.bashrc 下编辑

```sh
export JAVA_HOME=/home/bigdata/jdk1.8
export SCALA_HOME=/home/bigdata/scala-2.11.12
export PATH=$SCALA_HOME/bin:$JAVA_HOME/bin:$PATH
```

保存后，重新登入用户则生效

## 安装zookeeper

版本：3.4.14

解压

```sh
tar -xf zookeeper-3.4.14.tar.gz
cd zookeeper-3.4.14/
/home/bigdata/zookeeper-3.4.14 # pwd
cd conf/
cp zoo_sample.cfg zoo.cfg
```

修改zoo.cfg 并在/home/bigdata/zookeeper-3.4.14创建data目录

```sh
server.1=ict50:2888:3888
server.2=ict51:2888:3888
server.3=ict52:2888:3888
dataDir=/home/bigdata/zookeeper-3.4.14/data
```

在/home/bigdata/zookeeper-3.4.14/data下创建一个文件 myid，并在其中填入该服务器的序号比如ict50上为1

然后将zookeeper-3.4.14整体发给ict51 ict52 ，同时不要忘了修改相应的myid序号为2 3

```sh
scp -r zookeeper-3.4.14/ ict51:~/
scp -r zookeeper-3.4.14/ ict52:~/
```

然后运行启动/查看停止命令

```sh
[bigdata@ict51 zookeeper-3.4.14]$ ./bin/zkServer.sh start
ZooKeeper JMX enabled by default
Using config: /home/bigdata/zookeeper-3.4.14/bin/../conf/zoo.cfg
Starting zookeeper ... STARTED
[bigdata@ict51 zookeeper-3.4.14]$ ./bin/zkServer.sh status
ZooKeeper JMX enabled by default
Using config: /home/bigdata/zookeeper-3.4.14/bin/../conf/zoo.cfg
Mode: leader
```

可以看到ict51变为了leader

自此zookeepr部署完毕，可以根据需要将zookeeper注册为开机自启动

## 配置Hadoop文件

解压相应文件目录，此处我们解压到bigdata用户下

```sh
tar -xf hadoop-2.7.7.tar.gz
/home/bigdata/hadoop-2.7.7 # pwd
cd etc/hadoop
```

1. 修改hadoop-env.sh

   ```xml
   export JAVA_HOME=/home/bigdata/jdk1.8
   export HADOOP_PID_DIR=/home/hadoop/software/hadoop-2.7.7/tmp/pid/
   export HADOOP_SECURE_DN_PID_DIR=${HADOOP_PID_DIR}
   ```

2.  修改mapred-site.xml

   ```sh
   cp mapred-site.xml.template mapred-site.xml
   vim mapred-site.xml
   ```

   ```xml
   <?xml version="1.0"?>
   <?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
   <!--
     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License. See accompanying LICENSE file.
   -->
   
   <!-- Put site-specific property overrides in this file. -->
   
   <configuration>
     <!-- yarn manager framework -->
     <property>
       <name>mapreduce.framework.name</name>
       <value>yarn</value>
     </property>
     <!-- job history logs -->
     <property>
       <name>mapreduce.jobhistory.address</name>
       <value>ict51:10020</value>
     </property>
     <!-- job history logs webui-->
     <property>
        <name>mapreduce.jobhistory.webapp.address</name>
        <value>ict51:19888</value>
     </property>
   </configuration>
   ```

3. 修改yarn-site.xml

   ```sh
   vim yarn-site.xml
   ```

   ```xml
   <?xml version="1.0"?>
   <!--
     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License. See accompanying LICENSE file.
   -->
   <configuration>
     <property>
       <name>yarn.log-aggregation-enable</name>
       <value>true</value>
     </property>
     <property>
       <name>yarn.log-aggregation.retain-seconds</name>
       <value>259200</value>
     </property>
     <property>
       <name>yarn.log.server.url</name>
       <value>http://ict51:19888/jobhistory/logs</value>
     </property>
     <property>
       <name>yarn.resourcemanager.hostname</name>
       <value>ict50</value>
     </property>
     <property>
       <name>yarn.nodemanager.aux-services</name>
       <value>mapreduce_shuffle</value>
     </property>
       <property>
       <name>yarn.nodemanager.vmem-check-enabled</name>
       <value>false</value>
     </property>
     <property>
       <name>yarn.nodemanager.pmem-check-enabled</name>
       <value>false</value>
     </property>
   </configuration>
   ```
   
4. 修改core-site.xml 首先在hadoop目录下创建一个 tmp

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
   <!--
     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License. See accompanying LICENSE file.
   -->
   
   <!-- Put site-specific property overrides in this file. -->
   
   <configuration>
     <!-- namespace of HDFS, same as what in hdfs-site.xml -->
     <property>
       <name>fs.defaultFS</name>
       <value>hdfs://ict50:9000</value>
     </property>
     <!-- file's path of HDFS in hadoop-->
     <property>
       <name>hadoop.tmp.dir</name>
       <value>/home/bigdata/hadoop-2.7.7/tmp</value>
     </property>
   </configuration>
   ```
   
5. 修改hdfs-site.xml 在hadoop文件夹下创建hdfs存放本地目录

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
   <!--
     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at
   
       http://www.apache.org/licenses/LICENSE-2.0
   
     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License. See accompanying LICENSE file.
   -->
   
   <!-- Put site-specific property overrides in this file. -->
   
   <configuration>
     <property>
       <name>dfs.namenode.secondary.http-address</name>
       <value>ict52:50090</value>
     </property>
     <property>
       <name>dfs.replication</name>
       <value>3</value>
     </property>
     <property>
       <name>dfs.namenode.name.dir</name>
       <value>file:/home/bigdata/hadoop-2.7.7/hdfs/namenode</value>
     </property>
     <property>
       <name>dfs.datanode.data.dir</name>
       <value>file:/home/bigdata/hadoop-2.7.7/hdfs/datanode</value>
     </property>
   </configuration>
   ```
   
6. 修改slaves文件

   ```
   ict51
   ict52
   ```

7. 拷贝发送其他节点

   ```sh
   scp -r hadoop-2.7.7/ ict51:~/
   scp -r hadoop-2.7.7/ ict52:~/
   ```

## 启动

1. 首先启动zk节点

3. 在ict50上首次启动 需要格式化namenode

   ```
   ./bin/hdfs namenode -format
   显示如下信息即为成功
   20/04/28 16:00:18 INFO common.Storage: Storage directory /home/bigdata/hadoop-2.7.7/tmp/dfs/name has been successfully formatted.
   20/04/28 16:00:19 INFO namenode.FSImageFormatProtobuf: Saving image file /home/bigdata/hadoop-2.7.7/tmp/dfs/name/current/fsimage.ckpt_0000000000000000000 using no compression
   20/04/28 16:00:19 INFO namenode.FSImageFormatProtobuf: Image file /home/bigdata/hadoop-2.7.7/tmp/dfs/name/current/fsimage.ckpt_0000000000000000000 of size 324 bytes saved in 0 seconds.
   20/04/28 16:00:19 INFO namenode.NNStorageRetentionManager: Going to retain 1 images with txid >= 0
   20/04/28 16:00:19 INFO util.ExitUtil: Exiting with status 0
   20/04/28 16:00:19 INFO namenode.NameNode: SHUTDOWN_MSG: 
   /************************************************************
   SHUTDOWN_MSG: Shutting down NameNode at ict50/10.20.8.50
   ************************************************************/
   ```

5. 启动dfs

   ```sh
   ./sbin/start-dfs.sh
   当ict50 jps 出现namenode ict51 ict52出现datanode 说明启动成功
   失败可以先查看vim /etc/ssh/ssh_config 添加如下
   StrictHostKeyChecking no
   UserKnownHostsFile /dev/null
   ```

6. 启动yarn

   ```sh
   ict50:./sbin/start-yarn.sh
   ```
   
7. 启动historyserver 在ict51上

   ```sh
   ./sbin/mr-jobhistory-daemon.sh start historyserver
   ```
   
6. 这样就启动完成了，可以在web端查看各个页面

   ```
   haddop webui:http://10.20.8.50:50070
   mapreduce webui:http://10.20.8.50:8088
   ```

   

