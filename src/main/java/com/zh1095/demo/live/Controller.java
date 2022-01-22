package com.zh1095.demo.live;

import com.zh1095.demo.live.model.*;
import org.springframework.web.servlet.support.RequestContext;

/**
 * 直播
 *
 * <p>发起预约
 *
 * <p>获取通知，站内信
 *
 * @author zucker.1095
 */
public class Controller {
  /**
   * 用户预约直播 / 关注创作者 / 订阅合集
   *
   * <p>增加直播和用户双向关系
   *
   * @param ctx the ctx
   * @param req the req
   * @return add schedule rsp
   */
  public AddScheduleRsp AddSchedule(RequestContext ctx, AddScheduleReq req) {
    AddScheduleRsp rsp = new AddScheduleRsp();
    return rsp;
  }

  /**
   * 用户查看站内信
   *
   * <p>pull
   *
   * <p>push
   *
   * @param ctx the ctx
   * @param req the req
   * @return the get notification rsp
   */
  public GetNotificationRsp GetNotification(RequestContext ctx, GetNotificationReq req) {
    GetNotificationRsp rsp = new GetNotificationRsp();
    return rsp;
  }

  /**
   * 主播发送通知 & 供运营侧发系统通知
   *
   * <p>同步方式与 GetNotification 对齐
   *
   * <p>pull
   *
   * <p>push
   *
   * @param ctx the ctx
   * @param req the req
   * @return the send notification rsp
   */
  public SendNotificationRsp SendNotification(RequestContext ctx, SendNotificationReq req) {
    SendNotificationRsp rsp = new SendNotificationRsp();
    return rsp;
  }
}

