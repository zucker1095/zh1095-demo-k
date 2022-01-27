package com.zh1095.demo.live.clients.message;

import com.zh1095.demo.live.model.Notification;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Controller {}

class Plan1 {
  // as followed
  private final int uid;
  // 微博类的 feed 流单向，而朋友圈则双向，直播关注为前者，因此区分 from -> to
  private final List<Integer> fromIDs = new ArrayList<>(), toIDs = new ArrayList<>();
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
  public List<Notification> acceptPull(int page) {
    List<Notification> res = new ArrayList<>();
    // 是否有下一页内容
    boolean hasMore;
    // 下一页起始游标
    int nextCursor;
    // bottleneck1 O(n)，即使有缓存，如何更新
    reloadFollowerIDs();
    // bottleneck2，活跃用户
    // 并发拉取 & 塞入 res
    for (int toID : toIDs) {
      // bottleneck3，深分页
      Notification notification = Inbox.read(toID, page);
      res.add(notification);
    }
    // bottleneck4，排序
    res.sort((o1, o2) -> o1.createdTimeStamp > o2.createdTimeStamp ? 1 : -1);
    return res;
  }

  private void reloadFollowerIDs() {}

  /**
   * 保证发送成功，写扩散
   *
   * <p>需要维护两个 inbox
   *
   * @param notification
   */
  public void sendPush(Notification notification) {
    // bottleneck1 O(n)
    reloadFollowedIDs();
    // bottleneck2
    Inbox.write(uid, notification);
    // 并发发送
    for (int fromID : fromIDs) {
      Inbox.write(fromID, notification);
    }
  }

  private void reloadFollowedIDs() {}

  /**
   * 查看个人 timeline，即 user inbox，写扩散
   *
   * @return
   */
  public List<Notification> acceptPush(int page) {
    // bottleneck1
    List<Notification> res = Inbox.read(uid, page);
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
