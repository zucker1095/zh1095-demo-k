package com.zh1095.demo.improved.ggrammar;

/**
 * The interface F function.
 *
 * @author cenghui
 */
interface FFunction1 {
  /** Do it. */
  void doThis();
}

/** The interface F function 2. */
interface FFunction2 {
  /**
   * Do that int.
   *
   * @param num the name
   * @return the int
   */
  int doThat(int num);
}

/** The interface F function 3. */
interface FFunction3 {
  /** Do this and that. */
  void doThisAndThat();
}

/**
 * The type C class 1.
 *
 * @author cenghui
 */
class CClass1 implements FFunction2 {

  @Override
  public int doThat(int name) {
    return 0;
  }
}

/**
 * The type C class.
 *
 * @author cenghui
 */
class CClass2 implements FFunction1, FFunction2 {
  @Override
  public void doThis() {}

  @Override
  public int doThat(int num) {
    return 0;
  }
}

/**
 * Java7 引入匿名类
 *
 * <p>Method Reference System.out:print
 *
 * @author cenghui
 */
public class AAnonymous {
  /** Instantiates a new A anonymous. */
  public AAnonymous() {
    new CClass1().doThat(1);
  }
}

/**
 * Java8 提供 lambada expression，两个目的
 *
 * <p>以简化 Anonymous Class
 *
 * <p>实现 Function Programming
 *
 * @author cenghui
 */
class LLambada {}
