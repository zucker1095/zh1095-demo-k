package com.zh1095.demo.live.clients.message;

import com.zh1095.demo.live.model.Notification;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Controller {}

class Plan1 {
  // as followed
  private final int uid;
  private final List<Integer> followedIDs = new ArrayList<>(), followerIDs = new ArrayList<>();
  private final List<ThreadPoolExecutor> avaPool = new ArrayList<>();

  Plan1(int uid) {
    this.uid = uid;
  }

  private ThreadPoolExecutor newThreadPool(int cps, int mps, int type) {
    return new ThreadPoolExecutor(
        cps,
        mps,
        1000,
        TimeUnit.MILLISECONDS,
        type == 0 ? new ArrayBlockingQueue<>(cps) : new SynchronousQueue<>());
  }

  /**
   * 保证发送成功，读扩散
   *
   * @param notification
   */
  public void sendPull(Notification notification) {
    Inbox.write(uid, notification);
  }

  /**
   * 查看个人 timeline，即 user inbox，读扩散
   *
   * @return
   */
  public List<Notification> acceptPull() {
    List<Notification> res = new ArrayList<>();
    int len = followerIDs.size();
    // bottleneck1 O(n)
    reloadFollowerIDs();
    // bottleneck2
    // 并发拉取 & 塞入 res
    ThreadPoolExecutor pool = newThreadPool(len / 4, len / 2, 0);
    for (int followerID : followerIDs) {
      Runnable task =
          () -> {
            Notification notification = Inbox.read(followerID);
            synchronized (res) {
              res.add(notification);
            }
          };
      pool.execute(task);
    }
    pool.shutdown();
    return res;
  }

  private void reloadFollowerIDs() {}

  /**
   * 保证发送成功，写扩散
   *
   * @param notification
   */
  public void sendPush(Notification notification) {
    // bottleneck1 O(n)
    reloadFollowedIDs();
    // bottleneck2
    // 并发发送
    int len = followedIDs.size();
    ThreadPoolExecutor pool = newThreadPool(len / 4, len / 2, 1);
    for (int followedID : followedIDs) {
      pool.execute(
          () -> {
            Inbox.write(followedID, notification);
          });
    }
    pool.shutdown();
  }

  private void reloadFollowedIDs() {}

  /**
   * 查看个人 timeline，即 user inbox，写扩散
   *
   * @return
   */
  public List<Notification> acceptPush() {
    List<Notification> res = Inbox.read(uid);
    return res;
  }
}

class Plan2 extends Plan1 {
  Plan2(int uid) {
    super(uid);
  }

  @Override
  public List<Notification> acceptPull() {
    return super.acceptPull();
  }
}
