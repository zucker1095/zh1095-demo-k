package com.zh1095.demo.improved.io;

import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

public class Event {
  public static void main(String[] args) {
    List<Socket> sockets = new ArrayList<>();
  }

  // 事件派发
  // https://www.zhihu.com/question/445416579
  //  static int aeApiPoll(aeEventLoop *eventLoop, struct timeval *tvp) {
  //    // int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
  //    aeApiState *state = eventLoop->apidata;
  //    int retval, numevents = 0;
  //    retval = epoll_wait(state->epfd,state->events,eventLoop->setsize,
  //            tvp ? (tvp->tv_sec*1000 + tvp->tv_usec/1000) : -1);
  //    if (retval > 0) {
  //      int j;
  //      numevents = retval;
  //      for (j = 0; j < numevents; j++) {
  //        int mask = 0;
  //        struct epoll_event *e = state->events+j;
  //        if (e->events & EPOLLIN) mask |= AE_READABLE;
  //        if (e->events & EPOLLOUT) mask |= AE_WRITABLE;
  //        if (e->events & EPOLLERR) mask |= AE_WRITABLE|AE_READABLE;
  //        if (e->events & EPOLLHUP) mask |= AE_WRITABLE|AE_READABLE;
  //        eventLoop->fired[j].fd = e->data.fd;
  //        eventLoop->fired[j].mask = mask;
  //      }
  //    }
  //    return numevents;
  //  }

  private void handler() {}
}
