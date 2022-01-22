package com.zh1095.demo.live.clients.message;

import com.zh1095.demo.live.model.Notification;

public class Controller implements MessageImpl {
  private final MessageImpl messageImpl;

  Controller(int type) {
    messageImpl = type == 0 ? new PushMessage() : new PullMessage();
  }

  @Override
  public Notification accept() {
    return messageImpl.accept();
  }

  @Override
  public boolean send() {
    return messageImpl.send();
  }
}

interface MessageImpl {
  Notification accept();

  boolean send();
}

class PushMessage implements MessageImpl {
  @Override
  public Notification accept() {
    return null;
  }

  @Override
  public boolean send() {
    return false;
  }
}

class PullMessage implements MessageImpl {

  @Override
  public Notification accept() {
    return null;
  }

  @Override
  public boolean send() {
    return false;
  }
}
