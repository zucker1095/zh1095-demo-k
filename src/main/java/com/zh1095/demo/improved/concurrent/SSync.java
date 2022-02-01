package com.zh1095.demo.improved.concurrent;

public class SSync {
  public static void main(String[] args) {
    synchronized (SSync.class) {
    }
    m();
  }

  public static synchronized void m() {}
}
