package com.zh1095.demo.k.clients.zh1095_demo;

import org.apache.rocketmq.client.exception.MQBrokerException;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.client.producer.DefaultMQProducer;
import org.apache.rocketmq.client.producer.SendCallback;
import org.apache.rocketmq.client.producer.SendResult;
import org.apache.rocketmq.common.message.Message;
import org.apache.rocketmq.remoting.common.RemotingHelper;
import org.apache.rocketmq.remoting.exception.RemotingException;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class Rt1Producer extends Rt1 {
  // Message Size < 512k
  // producer is thread-safe
  // default timeout is 3000 milliseconds
  // SendStatus
  public static void main(String[] args)
      throws MQBrokerException, RemotingException, UnsupportedEncodingException,
          InterruptedException, MQClientException {
    produce();
    // produceOrderly();
  }

  private static DefaultMQProducer producer = null;

  Rt1Producer() throws MQClientException {
    // Instantiate with a producer group name.
    producer = new DefaultMQProducer(Rt1ProducerGroupName);
    // Specify name server addresses.
    producer.setNamesrvAddr(Rt1NamesrvAddr);
    // Launch the instance.
    producer.start();
  }

  private static void produce()
      throws UnsupportedEncodingException, MQBrokerException, RemotingException,
          InterruptedException, MQClientException {
    for (int i = 0; i < 100; i++) {
      // Create a message instance, specifying topic, tag and message body.
      Message msg =
          new Message(
              Rt1TopicName, TagA, ("Hello RocketMQ " + i).getBytes(RemotingHelper.DEFAULT_CHARSET));
      // Call send message to deliver message to one of brokers.
      SendResult res = producer.send(msg);
      System.out.printf("%s%n", res);
    }
    // Shut down once the producer instance is not longer in use.
    producer.shutdown();
  }

  private record SendCallbacker(CountDownLatch countDownLatch, int index) implements SendCallback {

    @Override
      public void onSuccess(SendResult res) {
      countDownLatch.countDown();
      System.out.printf("%-10d OK %s %n", index, res.getMsgId());
      }

      @Override
      public void onException(Throwable e) {
        countDownLatch.countDown();
        System.out.printf("%-10d Exception %s %n", index, e);
        e.printStackTrace();
      }
  }

  private static void produceAsync() throws InterruptedException {
    producer.setRetryTimesWhenSendAsyncFailed(0);
    int messageCount = 100;
    CountDownLatch countDownLatch = new CountDownLatch(messageCount);
    for (int i = 0; i < messageCount; i++) {
      try {
        final int index = i;
        Message msg =
            new Message(
                Rt1TopicName,
                TagA,
                "OrderID188",
                "Hello world".getBytes(RemotingHelper.DEFAULT_CHARSET));
        producer.send(msg, new SendCallbacker(countDownLatch, index));
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    countDownLatch.await(5, TimeUnit.SECONDS);
    producer.shutdown();
  }

  private static void produceOrderly()
      throws MQClientException, UnsupportedEncodingException, MQBrokerException, RemotingException,
          InterruptedException {
    String[] tags = new String[] {TagA, TagB};
    for (int i = 0; i < 100; i++) {
      int orderId = i % 10;
      // Create a message instance, specifying topic, tag and message body.
      Message msg =
          new Message(
              Rt1TopicName,
              tags[i % tags.length],
              "KEY" + i,
              ("Hello RocketMQ " + i).getBytes(RemotingHelper.DEFAULT_CHARSET));
      SendResult res =
          producer.send(
              msg,
              (mqs, msg1, arg) -> {
                Integer id = (Integer) arg;
                return mqs.get(id % mqs.size());
              },
              orderId);
      System.out.printf("%s%n", res);
    }
    // server shutdown
    producer.shutdown();
  }

  // 建议总体积 < 512k
  private static void produceBatch() {
    List<Message> messages = new ArrayList<>();
    messages.add(new Message(Rt1TopicName, TagA, "OrderID001", "Hello world 0".getBytes()));
    messages.add(new Message(Rt1TopicName, TagA, "OrderID002", "Hello world 1".getBytes()));
    messages.add(new Message(Rt1TopicName, TagA, "OrderID003", "Hello world 2".getBytes()));
    try {
      producer.send(messages);
    } catch (Exception e) {
      e.printStackTrace();
      // handle the error
    }
  }
}
