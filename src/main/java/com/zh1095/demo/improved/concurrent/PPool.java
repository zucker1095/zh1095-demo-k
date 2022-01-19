package com.zh1095.demo.improved.concurrent;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class PPool {
  public static void main(String[] args) {
    //
  }

  private static void test1() {
    // 140 量级
    ThreadPoolExecutor pool1 =
        new ThreadPoolExecutor(
            70, 120, 500, TimeUnit.MILLISECONDS, new ArrayBlockingQueue<Runnable>(80));

    pool1.execute(() -> {});

    pool1.submit(
        () -> {
          return 1;
        });

    pool1.shutdown();
  }
}
