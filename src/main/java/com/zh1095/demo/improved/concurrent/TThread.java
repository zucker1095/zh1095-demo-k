package com.zh1095.demo.improved.concurrent;

import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class TThread {
  public static void main(String[] args) {
    test1();
  }
  // 1.两种启线程方式，建议 Runnable
  private static void test1() {
    try {
      // 2.因为 Runnable 是 interface 而 Thread 为 class
      Runnable t1 =
          () -> {
            System.out.println("runnable" + Thread.currentThread().getName());
          };
      // thread.start() 即调用 runnable.run()
      Thread t2 =
          new Thread(
              () -> {
                System.out.println("thread" + Thread.currentThread().getName());
              });
      t1.run();
      t2.start();
      // 3.休眠二者等价，建议前者
      TimeUnit.MILLISECONDS.sleep(500);
      // Thread.sleep(1000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    System.out.println("Done!");
  }

  private static void test2() {}
}
