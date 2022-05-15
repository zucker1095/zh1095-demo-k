package com.zh1095.demo.improved;

public class Main {
  public static void main(String[] args) {
    testList(new int[] {8, 3, 6, 5});

    testList(new int[] {8, 3, 6});
  }

  private static void testList(int[] tests2) {
    ListNode h = new ListNode(1), cur = h;
    for (int n : tests2) {
      cur.next = new ListNode(n);
      cur = cur.next;
    }
    h = sortOddEvenList(h);
    while (h.next != null) {
      System.out.println(h.val);
      h = h.next;
    }
  }

  /**
   * 重排奇偶链表，奇数位置升序，偶数位置反之，升序排列整个链表
   *
   * <p>分别取出奇偶链表，奇数位链表需断尾 & 反转偶数位链表 & 合并
   *
   * <p>参考 https://mp.weixin.qq.com/s/0WVa2wIAeG0nYnVndZiEXQ
   *
   * @param head
   * @return
   */
  static ListNode sortOddEvenList(ListNode head) {
    return mergeTwoLists(head, separateOddEvenList(head));
  }

  static ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    ListNode dummy = new ListNode(), cur = dummy, l1 = list1, l2 = list2;
    while (l1 != null && l2 != null) {
      if (l1.val < l2.val) {
        cur.next = l1;
        l1 = l1.next;
      } else {
        cur.next = l2;
        l2 = l2.next;
      }
      cur = cur.next;
    }
    cur.next = l1 != null ? l1 : l2;
    return dummy.next;
  }

  // 代码与「奇偶链表」一致，但分离且返回偶头
  static ListNode separateOddEvenList(ListNode head) {
    if (head == null) return null;
    ListNode odd = head, even = head.next, nxtOdd = null;
    if (even != null) {
      nxtOdd = even.next;
      even.next = null;
    }
    while (nxtOdd != null) {
      nxtOdd = even.next;
      odd.next = nxtOdd;
      odd = odd.next;
      if (nxtOdd.next == null) break;
      nxtOdd = nxtOdd.next.next;
      nxtOdd.next.next = even;
      even = nxtOdd.next;
    }
    odd.next = null;
    return even;
  }

  static class ListNode { // 链表节点
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
}
