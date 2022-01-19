package com.zh1095.demo.improved.cconcurreny;

import java.util.LinkedList;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * 生产者消费者模式
 *
 * @author cenghui
 */
public class CConsume {}

/**
 * 基于阻塞队列实现，内部维护两个条件变量以保证并发安全，本质即 wait/notify
 *
 * <p>参考 Go channel 的实现
 *
 * @param <T> the type parameter
 */
class BBlockingQueue<T> {
  private final LinkedList<T> elements = new LinkedList<>();
  private final int capacity;
  private final Lock lock = new ReentrantLock(); // 保证 elements 互斥访问
  private final Condition notFull = lock.newCondition(); // 通知不同的角色
  private final Condition notEmpty = lock.newCondition();

  /**
   * Instantiates a new B blocking queue.
   *
   * @param capacity the capacity
   */
  public BBlockingQueue(int capacity) {
    this.capacity = capacity;
  }

  /**
   * Put.
   *
   * @param element the element
   * @throws InterruptedException the interrupted exception
   */
  public void put(T element) throws InterruptedException {
    lock.lock();
    try {
      while (elements.size() == capacity) notFull.await();
      elements.add(element);
      notEmpty.signalAll();
    } finally {
      lock.unlock();
    }
  }

  /**
   * Take t.
   *
   * @return the t
   * @throws InterruptedException the interrupted exception
   */
  public T take() throws InterruptedException {
    lock.lock();
    try {
      while (elements.isEmpty()) notEmpty.await();
      T first = elements.pollFirst();
      notFull.signalAll();
      return first;
    } finally {
      lock.unlock();
    }
  }
}

/** 两个线程交替打印奇偶 */
class TTest1 {
  private int num = 1;
  private final Lock lock = new ReentrantLock();
  private final Condition cond = lock.newCondition();

  public TTest1() {
    new Thread(
            () -> {
              while (true) {
                lock.lock();
                try {
                  if (num % 2 == 0) cond.await();
                  System.out.println(num + "odd");
                  num += 1;
                  if (num > 11) return;
                  cond.signalAll();
                } catch (InterruptedException e) {
                  e.printStackTrace();
                } finally {
                  lock.unlock();
                }
              }
            })
        .start();
    new Thread(
            () -> {
              while (true) {
                lock.lock();
                try {
                  if (num % 2 == 1) cond.await();
                  System.out.println(num + "even");
                  num += 1;
                  if (num > 10) return;
                  cond.signalAll();
                } catch (InterruptedException e) {
                  e.printStackTrace();
                } finally {
                  lock.unlock();
                }
              }
            })
        .start();
  }
}

/** 生产者消费者模式 */
class TTest2 {
  public TTest2() {
    BBlockingQueue<Integer> bq = new BBlockingQueue<Integer>(2);
    new Thread(
            () -> {
              for (int i = 0; i < 3; i++) {
                try {
                  bq.put(i);
                  System.out.println(i + "produce");
                } catch (InterruptedException e) {
                  e.printStackTrace();
                }
              }
            })
        .start();
    new Thread(
            () -> {
              for (int i = 0; i < 3; i++) {
                try {
                  int num = bq.take();
                  System.out.println(num + "consume");
                } catch (InterruptedException e) {
                  e.printStackTrace();
                }
              }
            })
        .start();
  }
}
