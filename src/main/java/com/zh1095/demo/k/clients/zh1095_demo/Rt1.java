package com.zh1095.demo.k.clients.zh1095_demo;

public abstract class Rt1 {
  protected static final String Rt1NamesrvAddr = "127.0.0.1:9876";

  protected static final String Rt1TopicName = "zh1095_demo_rt1";
  protected static final String Rt1ConsumerGroupName = "zh1095_demo_rt1_cg1";
  protected static final String Rt1ProducerGroupName = "zh1095_demo_rt1_pg1";


  protected static final String TagWildcard = "*";
  protected static final String TagA = "zh1095_demo_tagA";
  protected static final String TagB = "zh1095_demo_tagB";

  protected static String addTag(String desc, String src) {
    return desc + " || " + src;
  }
}
