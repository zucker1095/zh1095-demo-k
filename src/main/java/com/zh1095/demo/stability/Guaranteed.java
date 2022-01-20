package com.zh1095.demo.stability;

/**
 * 中台兜底优化 https://bytedance.feishu.cn/wiki/wikcnIvjs6U7mkHpNaEVO0LXhpc
 *
 * <p>situation 春节前需要保证稳定性，目前内存占用常态四成以上
 *
 * <p>task 查看代码，初步定位为
 *
 * <p>action 减少对中心存储的依赖
 *
 * <p>result 内存占用常态降至二成
 */
public class Guaranteed {
  /**
   * The entry point of application.
   *
   * @param args the input arguments
   */
  public static void main(String[] args) {
    plan1();
  }

  /**
   * 初次方案定
   *
   * <p>飞好吃
   */
  public static void plan1() {}
}
