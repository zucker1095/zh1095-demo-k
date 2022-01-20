package com.zh1095.demo.live;

import java.util.ArrayList;
import java.util.List;

/**
 * 模拟直播的消息推送流，包括
 *
 * <p>直播预约通知
 *
 * <p>直播
 *
 * <p>朋友圈 / Twitter 的设计方案，前者相当于互关的后者，消息的推送是相似的
 */
public class FeedClient {
  private final User user;

  /**
   * Instantiates a new Feed client.
   *
   * @param uid the uid
   */
  FeedClient(int uid) {
    this.user = new User(uid);
  }

  /**
   * 读扩散
   *
   * <p>1.查询关注用户
   *
   * <p>2.查询关注用户发布内容 / 获取更新状态
   *
   * @param targetUID the target uid
   * @return list list
   */
  public List<Feed> pull(int targetUID) {
    List<Feed> res = new ArrayList<>();
    return res;
  }

  /**
   * 写扩散
   *
   * <p>1.查询关注的 feeds 列表
   *
   * <p>2.逐个推送
   */
  public void push() {}

  /**
   * 混合
   *
   * <p>活跃用户使用 push，由于活跃用户关注数多，pull 效率低，且触发 feeds 请求频次较高 发布内容时只对粉丝中的活跃用户做推送，活跃用户直接 pull 关注的 feeds 列表
   *
   * <p>非活跃用户使用 pull，走正常的 1+N 两次查询
   */
  public void hybrid() {}
}

/** The type Feed. */
class Feed {
  // meta
  private int uid;
  private String title;
  private String content;
  // addition
  private String[] tags;
  private String topic;
}
