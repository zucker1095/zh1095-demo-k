package com.zh1095.demo.k.clients.zh1095_demo;

import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.*;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.common.consumer.ConsumeFromWhere;
import org.apache.rocketmq.common.message.MessageExt;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

public class Rt1Consumer extends Rt1 {
  // subscribe relation: consumer group 1->1 topic 1->n consumer group

  private static DefaultMQPushConsumer consumer = null;

  Rt1Consumer() {
    // Instantiate with specified consumer group name.
    consumer = new DefaultMQPushConsumer(Rt1ConsumerGroupName);
    // Specify name server addresses.
    consumer.setNamesrvAddr(Rt1NamesrvAddr);
    // Subscribe one more more topics to consume.
  }

  public static void main(String[] args) throws MQClientException {
    consume();
  }

  public static void consume() throws MQClientException {
    consumer.subscribe(Rt1TopicName, TagWildcard);
    // Register callback to execute on arrival of messages fetched from brokers.
    consumer.registerMessageListener(new MessageHandlerCon());
    // Launch the consumer instance.
    consumer.start();
    System.out.printf("Consumer Started.%n");
  }

  public static void consumeOrderly() throws MQClientException {
    consumer.setConsumeFromWhere(ConsumeFromWhere.CONSUME_FROM_FIRST_OFFSET);
    consumer.subscribe(Rt1TopicName, addTag(TagA, TagB));
    consumer.registerMessageListener(new MessageHandlerOrder());
    consumer.start();
    System.out.printf("Consumer Started.%n");
  }

  private static class MessageHandlerCon implements MessageListenerConcurrently {
    @Override
    public ConsumeConcurrentlyStatus consumeMessage(
        List<MessageExt> list, ConsumeConcurrentlyContext consumeConcurrentlyContext) {
      System.out.printf("%s Receive New Messages: %s %n", Thread.currentThread().getName(), list);
      return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
    }
  }

  private static class MessageHandlerOrder implements MessageListenerOrderly {
    final AtomicLong consumeTimes = new AtomicLong(0);

    @Override
    public ConsumeOrderlyStatus consumeMessage(
        List<MessageExt> msgs, ConsumeOrderlyContext context) {
      context.setAutoCommit(false);
      System.out.printf(Thread.currentThread().getName() + " Receive New Messages: " + msgs + "%n");
      this.consumeTimes.incrementAndGet();
      if ((this.consumeTimes.get() % 2) == 0) {
        return ConsumeOrderlyStatus.SUCCESS;
      } else if ((this.consumeTimes.get() % 3) == 0) {
        return ConsumeOrderlyStatus.ROLLBACK;
      } else if ((this.consumeTimes.get() % 4) == 0) {
        return ConsumeOrderlyStatus.COMMIT;
      } else if ((this.consumeTimes.get() % 5) == 0) {
        context.setSuspendCurrentQueueTimeMillis(3000);
        return ConsumeOrderlyStatus.SUSPEND_CURRENT_QUEUE_A_MOMENT;
      }
      return ConsumeOrderlyStatus.SUCCESS;
    }
  }
}
