# 利用maven建立Java项目

1. 首先创建一个maven工程

2. 修改pom文件

   

   ```xml
   <?xml version="1.0" encoding="UTF-8"?>
   <project xmlns="http://maven.apache.org/POM/4.0.0"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
       <modelVersion>4.0.0</modelVersion>
   
       <groupId>per.lihao</groupId>
       <artifactId>sparkt</artifactId>
       <version>1.0-SNAPSHOT</version>
   
       <properties>
           <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
           <spark.version>2.4.5</spark.version>
           <pgsql.version>10.1</pgsql.version>
           <!--<kafka.version></kafka.version>-->
       </properties>
       <dependencies>
           <dependency>
               <groupId>org.apache.spark</groupId>
               <artifactId>spark-core_2.12</artifactId>
               <version>${spark.version}</version>
               <scope>provided</scope>
           </dependency>
       </dependencies>
       <build>
           <plugins>
               <plugin>
                   <!--将依赖文件集中打包-->
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-assembly-plugin</artifactId>
                   <configuration>
                       <descriptorRefs>
                           <descriptorRef>jar-with-dependencies</descriptorRef>
                       </descriptorRefs>
                       <archive>
                           <manifest>
                               <mainClass>per.lihao.WordCount</mainClass>
                           </manifest>
                       </archive>
                   </configuration>
                   <executions>
                       <execution>
                           <id>make-assembly</id>
                           <phase>package</phase>
                           <goals>
                               <goal>single</goal>
                           </goals>
                       </execution>
                   </executions>
               </plugin>
   
               <plugin>
                   <groupId>org.apache.maven.plugins</groupId>
                   <artifactId>maven-compiler-plugin</artifactId>
                   <configuration>
                       <source>1.8</source>
                       <target>1.8</target>
                   </configuration>
               </plugin>
           </plugins>
       </build>
   </project>
   ```

3. 编写计算代码

   ```java
   package per.lihao;
   
   import org.apache.spark.SparkConf;
   import org.apache.spark.api.java.JavaPairRDD;
   import org.apache.spark.api.java.JavaSparkContext;
   import scala.Tuple2;
   
   import java.util.Arrays;
   import java.util.List;
   
   public class WordCount {
       public static void main(String[] args) {
           // 读取文件
           String filename = args[0];
           SparkConf sparkConf = new SparkConf().setAppName("WordCount").setMaster("local");
           JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
           JavaPairRDD<String, Integer> resultRDD = javaSparkContext.textFile(filename).
                   flatMap(s -> Arrays.asList(s.split(" ")).iterator())
                   // map to pair
                   .mapToPair(word -> new Tuple2<>(word, 1)).reduceByKey((a, b) -> a + b);
           // 倒序排序
           JavaPairRDD<Integer, String> sortsRDD = resultRDD.mapToPair(t -> new Tuple2<>(t._2, t._1)).sortByKey(false);
           List<Tuple2<Integer, String>> top10 = sortsRDD.take(10);
           top10.forEach(t -> System.out.println(t._2 + "\t" + t._1));
           // 关闭资源
           javaSparkContext.stop();
       }
   }
   
   ```

4. maven编译打包

   编译时 可以把spark依赖中的<scope>provided</scope>打开，但是在本地idea运行中 不能打开，否则spark相关类不会编译

   ![编译步骤](img\mvn-package.png) 

5. 上传至spark服务器，输入命令运行

   ![上传至spark服务器](img\upload-word-count-spark.png)

   ```sh
   ./bin/spark-submit \
   > --class per.lihao.WordCount \
   > --master local \
   > sparkt-1.0-SNAPSHOT-jar-with-dependencies.jar name.txt
   ```

   运行成功后返回

   ![wordcount结果](img\wordcount-result.png)

