package com.zh1095.demo.improved.aalgorithmn;

import java.util.ArrayList;
import java.util.List;

/**
 * The type List node.
 *
 * @author cenghui
 */
class ListNode { // 链表节点
  /** The Val. */
  int val;

  /** The Next. */
  ListNode next;

  /** Instantiates a new List node. */
  ListNode() {}

  /**
   * Instantiates a new List node.
   *
   * @param val the val
   */
  ListNode(int val) {
    this.val = val;
  }

  /**
   * Instantiates a new List node.
   *
   * @param val the val
   * @param next the next
   */
  ListNode(int val, ListNode next) {
    this.val = val;
    this.next = next;
  }
}

/**
 * The type Tree node.
 *
 * @author cenghui
 */
class TreeNode { // 树节点
  /** The Left. */
  TreeNode left,
      /** The Right. */
      right;

  /** The Val. */
  int val;

  /**
   * Instantiates a new Tree node.
   *
   * @param i the
   */
  TreeNode(int i) {}
}

/**
 * 面向面试，因此尽可能简单
 *
 * <p>参考 Go container/heap 编写
 *
 * <p>扩展性只需更改 less()
 *
 * @author cenghui
 */
class MinHeap {
  private final List<Integer> pq;
  /**
   * heapify from 首个非叶节点
   *
   * @param nums the nums
   */
  public MinHeap(int[] nums) {
    pq = new ArrayList<>(nums.length);
    for (int num : nums) pq.add(num);
    for (int i = (pq.size() >> 1) - 1; i >= 0; i--) down(i);
  }

  /**
   * push & up
   *
   * @param val the val
   */
  public void push(int val) {
    pq.add(val);
    up(pq.size() - 1);
  }

  /**
   * swap & down & poll
   *
   * @return the int
   */
  public int poll() {
    // 最大堆的堆顶就是最大元素
    int num = pq.get(0);
    // 把这个最大元素换到最后，删除之
    swap(0, pq.size() - 1);
    pq.remove(pq.size() - 1);
    // 让 pq[1] 下沉到正确位置
    down(0);
    return num;
  }

  /**
   * Peek int.
   *
   * @return the int
   */
  public int peek() {
    return pq.get(0);
  }

  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return pq.size();
  }

  private boolean less(int idx1, int idx2) {
    return pq.get(idx1) < pq.get(idx2);
  }

  private void swap(int a, int b) {
    int tmp = pq.get(a);
    pq.set(a, pq.get(b));
    pq.set(b, tmp);
  }

  private void up(int targetIdx) {
    int curIdx = targetIdx;
    while (true) {
      int parentIdx = (curIdx - 1) >> 1; // parent
      if (parentIdx == curIdx || less(curIdx, parentIdx)) break;
      swap(parentIdx, curIdx);
      curIdx = parentIdx;
    }
  }

  private boolean down(int targetIdx) {
    int curIdx = targetIdx;
    while (true) {
      int leftIdx = 2 * curIdx + 1;
      if (leftIdx >= pq.size() - 1 || leftIdx < 0) break; // leftIdx < 0 after int overflow
      // int j = leftIdx; // left child
      if (leftIdx + 1 < pq.size() - 1 && less(leftIdx + 1, leftIdx))
        leftIdx += 1; // = 2*i + 2  // right child
      if (less(leftIdx, curIdx)) break;
      swap(curIdx, leftIdx);
      curIdx = leftIdx;
    }
    return curIdx > targetIdx; // 是否 down 了
  }
}

/**
 * The type Data.
 *
 * @author cenghui
 */
public class Data {}
