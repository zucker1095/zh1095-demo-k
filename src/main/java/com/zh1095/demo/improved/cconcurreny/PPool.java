package com.zh1095.demo.improved.cconcurreny;

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/** 线程池相关 */
public class PPool {
  private void testing1() {
    ThreadPoolExecutor pool1 =
        new ThreadPoolExecutor(2, 4, 60L, TimeUnit.SECONDS, new SynchronousQueue<>());
    for (int i = 0; i < 3; i++) pool1.execute(() -> {});
  }

  private void testing2() {
    ThreadPoolExecutor pool2 =
        new ThreadPoolExecutor(2, 4, 60L, TimeUnit.SECONDS, new ArrayBlockingQueue<>(3));
    for (int i = 0; i < 3; i++) pool2.execute(() -> {});
  }

  private void testing3() {
    ThreadPoolExecutor pool3 =
        new ThreadPoolExecutor(2, 4, 60L, TimeUnit.SECONDS, new PriorityBlockingQueue<>(3));
    for (int i = 0; i < 3; i++) pool3.execute(() -> {});
  }
}

/** 实现线程池 */
class TThreadPoolExecutor {
  private final int corePoolSize;
  private final int maximumPoolSize;
  private final BlockingQueue<Runnable> blockingQueue;
  private final RejectedExecutionHandler rejectHandler;
  private final AtomicInteger ctl = new AtomicInteger();

  public TThreadPoolExecutor(
      int corePoolSize,
      int maximumPoolSize,
      BlockingQueue<Runnable> blockingQueue,
      RejectedExecutionHandler rejectHandler) {
    this.corePoolSize = corePoolSize;
    this.maximumPoolSize = maximumPoolSize;
    this.blockingQueue = blockingQueue;
    this.rejectHandler = rejectHandler;
  }

  public void execute(Runnable command) {
    if (ctl.get() >> 16 < corePoolSize) {
      new Thread(command).start();
    } else if (corePoolSize <= ctl.get() && ctl.get() < maximumPoolSize) {

    } else if (maximumPoolSize <= ctl.get()) {

    }
  }

  private void addWorker(Worker worker) {}

  private class Worker {}
}
