package com.zh1095.demo.improved.jvm;

/**
 * 用于分析字节码 javap -verbose SimpleClass.java
 *
 * <p>https://docs.oracle.com/javase/8/docs/technotes/tools/
 *
 * <p>相关命令行工具
 *
 * <p>java
 *
 * <p>javac
 *
 * <p>javap
 *
 * <p>jstack 堆栈跟踪工具，用来打印目标 Java 进程中各个线程的栈轨迹，以及这些线程所持有的锁，并可以生成 java 虚拟机当前时刻的线程快照
 *
 * <p>jstat 虚拟机统计信息监视工具，用于监视虚拟机运行时状态信息，它可以显示出虚拟机进程中的类装载、内存、垃圾收集、JIT 编译等运行数据
 */
public class CClassTest {
  private int m;

  public int inc() {
    return m + 1;
  }
}
