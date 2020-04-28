# Hadoop-HA集群部署

## 准备节点

将以下加入/etc/hosts

- 10.20.8.50 ict50
- 10.20.8.51 ict51
- 10.20.8.52 ict52

其中对于每个节点规划的功能如下

| 节点  | Hadoop相关功能                        | Yarn相关功能                        |
| ----- | :------------------------------------ | ----------------------------------- |
| ict50 | NameNode                              | ResourceManager                     |
| ict51 | DataNode,JournalNode                  | NodeManager,StandBy ResourceManager |
| ict52 | DataNode,JournalNode,SecondryNameNode | NodeManager                         |

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
systemctl disabled firewalld
```

## 配置JDK和Scala

JDK版本：1.8 （不要openjdk）

Scala版本：2.12.8

上传压缩包到bigdata用户下，解压

在~/.bashrc 下编辑

```sh
export JAVA_HOME=/home/bigdata/jdk1.8
export SCALA_HOME=/home/bigdata/scala-2.12.8
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

1. 修改hadoop-env.xml

   ```xml
   export JAVA_HOME=/home/bigdata/jdk1.8
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
     <!-- start ha-->
     <property>  
       <name>yarn.resourcemanager.ha.enabled</name>  
       <value>true</value>  
     </property>
     <!-- define ha cluster id -->
     <property>  
       <name>yarn.resourcemanager.cluster-id</name>  
       <value>gpos</value>  
     </property> 
     <!-- define ha resource manager ids -->
     <property>  
       <name>yarn.resourcemanager.ha.rm-ids</name>  
       <value>rm1,rm2</value>  
     </property>
     <property>  
       <name>yarn.resourcemanager.hostname.rm1</name>  
       <value>ict50</value>  
     </property> 
     <property>  
       <name>yarn.resourcemanager.hostname.rm2</name>  
       <value>ict51</value>  
     </property>
     <!-- merge mr -->
     <property>
       <name>yarn.nodemanager.aux-services</name>
       <value>mapreduce_shuffle</value>
     </property>
     <!-- if failure resourcemanager recovery -->
     <property>
       <name>yarn.resourcemanager.recovery.enabled</name>
       <value>true</value>
     </property>
     <!-- define zk recovery -->
     <property>
       <name>yarn.resourcemanager.store.class</name>
       <value>org.apache.hadoop.yarn.server.resourcemanager.recovery.ZKRMStateStore</value>
     </property>
     <property>
       <name>yarn.resourcemanager.zk-address</name>
       <value>ict50:2181,ict51:2181,ict52:2181</value>
     </property>
   </configuration>
   ```

4. 修改core-site.xml 首先在hadoop目录下创建一个 tmp，用来存放hdfs数据

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
       <value>hdfs://ns1</value>   
     </property>
     <!-- file's path of HDFS in hadoop-->
     <property>
       <name>hadoop.tmp.dir</name>              
       <value>/home/bigdata/hadoop-2.7.7/tmp</value>   
     </property>
     <!-- zk infos-->
     <property>
       <name>ha.zookeeper.quorum</name>              
       <value>ict50:2181,ict51:2181,ict52:2181</value>   
     </property>
   </configuration>
   ```

5. 修改hdfs-site.xml 在hadoop文件夹下创建journaldata存放日志本地目录

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
     <!-- namespace of HDFS, same as what in core-site.xml -->
     <property>
       <name>dfs.nameservices</name>              
       <value>ns1</value>   
     </property>
     <!-- NameNode in ns1 ,it is HA -->
     <property>
       <name>dfs.ha.namenodes.ns1</name>              
       <value>ict50,ict52</value>   
     </property>
     <!-- ict50 RPC  -->
     <property>
       <name>dfs.namenode.rpc-address.ns1.ict50</name>              
       <value>ict50:9000</value>   
     </property>
     <!-- ict50 webui -->
     <property>
       <name>dfs.namenode.http-address.ns1.ict50</name>              
       <value>ict50:50070</value>   
     </property>
     <!-- ict52 RPC  -->
     <property>
       <name>dfs.namenode.rpc-address.ns1.ict52</name>              
       <value>ict52:9000</value>   
     </property>
     <!-- ict52 webui -->
     <property>
       <name>dfs.namenode.http-address.ns1.ict52</name>              
       <value>ict52:50070</value>   
     </property>
     <!-- namenode metadata location in hdfs -->
     <property>
       <name>dfs.namenode.shared.edits.dir</name>              
       <value>qjournal://ict51:8485;ict52:8485/ns1</value>   
     </property>
     <!-- journalnode data in local -->
     <property>
       <name>dfs.journalnode.edits.dir</name>              
       <value>/home/bigdata/hadoop-2.7.7/journaldata</value>   
     </property>
     <!-- namenode ha -->
     <property>
       <name>dfs.ha.automatic-failover.enabled</name>              
       <value>true</value>   
     </property>
     <property>
       <name>dfs.client.failover.proxy.provider.ns1</name>              
       <value>org.apache.hadoop.hdfs.server.namenode.ha.ConfiguredFailoverProxyProvider</value>   
     </property>
     <property>
       <name>dfs.ha.fencing.methods</name>              
       <value>
         sshfencing
         shell(bin/true)
       </value>   
     </property>
     <property>
       <name>dfs.ha.fencing.ssh.private-key-files</name>              
       <value>/home/bigdata/.ssh</value>   
     </property>
     <property>
       <name>dfs.ha.fencing.ssh.connect-timeout</name>              
       <value>30000</value>   
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

2. 启动ict51 ict52上的journalnode 

   ```sh
   sbin/hadoop-daemon.sh start journalnode
   ```

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

4. 格式化zkfc服务

   ```sh
   ./bin/hdfs zkfc -formatZK
   出现如下信息即为成功
   20/04/28 16:02:17 INFO zookeeper.ClientCnxn: Opening socket connection to server ict52/10.20.8.52:2181. Will not attempt to authenticate using SASL (unknown error)
   20/04/28 16:02:17 INFO zookeeper.ClientCnxn: Socket connection established to ict52/10.20.8.52:2181, initiating session
   20/04/28 16:02:17 INFO zookeeper.ClientCnxn: Session establishment complete on server ict52/10.20.8.52:2181, sessionid = 0x3000a8751520000, negotiated timeout = 5000
   20/04/28 16:02:17 INFO ha.ActiveStandbyElector: Session connected.
   20/04/28 16:02:17 INFO ha.ActiveStandbyElector: Successfully created /hadoop-ha/ns1 in ZK.
   20/04/28 16:02:17 INFO zookeeper.ZooKeeper: Session: 0x3000a8751520000 closed
   20/04/28 16:02:17 INFO zookeeper.ClientCnxn: EventThread shut down
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
   ict51:./sbin/yarn-daemon.sh start resourcemanager
   ```

7. 这样就启动完成了，可以在web端查看各个页面

   ```
   haddop webui:http://10.20.8.50:50070
   
   ```

   此HA方法有待于完善

