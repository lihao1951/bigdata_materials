# Maven踩坑

## 在maven中需要配置国内的中央仓库

settings.xml如下

```xml
    <localRepository>D:/applications/apache-maven-3.6.0/m2</localRepository>

    <mirror>    
      <id>nexus-aliyun</id>  
      <name>nexus-aliyun</name>
      <url>https://maven.aliyun.com/nexus/content/groups/public</url>  
      <mirrorOf>central</mirrorOf>    
    </mirror>
```

其中 mirrorOf 不能配置*，且同样的名称的mirrorOf 只有第一个起作用

当需要第三方的库时，可以在pom文件中加入

```xml
    <!-- 添加confluent官方maven仓库 -->
    <repositories>
        <repository>
            <id>confluent</id>
            <url>https://packages.confluent.io/maven/</url>
        </repository>
    </repositories>
```

## IDEA 中的Maven配置

需要点击关闭 Work Offline

