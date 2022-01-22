package com.zh1095.demo.improved.algorithmn;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * 收集所有链表相关
 *
 * <p>尾插 & 暂存 & 变向 & 步进
 *
 * <p>反转 & 环形 & 回文 & 合并 & 删除
 *
 * @author cenghui
 */
public class LList {
  protected int getVal(ListNode node) {
    return node == null ? 0 : node.val;
  }

  /**
   * 反转链表，三步曲，暂存 & 变向 & 步进
   *
   * @param head the head
   * @return the list node
   */
  public ListNode reverseList(ListNode head) {
    ListNode pre = null, cur = head;
    while (cur != null) {
      ListNode nxt = cur.next; // 1.暂存
      cur.next = pre; // 2.变向
      pre = cur; // 3.步进
      cur = nxt;
    }
    return pre;
  }
  /**
   * 合并两个有序链表，正向，参考合并两个有序数组
   *
   * <p>模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * @param list1 the list 1
   * @param list2 the list 2
   * @return the list node
   */
  public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    ListNode dummy = new ListNode();
    ListNode cur = dummy, l1 = list1, l2 = list2;
    while (l1 != null && l2 != null) {
      if (l1.val <= l2.val) {
        cur.next = l1;
        l1 = l1.next;
      } else {
        cur.next = l2;
        l2 = l2.next;
      }
      cur = cur.next;
    }
    if (l1 != null) cur.next = l1;
    else if (l2 != null) cur.next = l2;
    return dummy.next;
  }
  /**
   * 链表的中间结点，快慢指针
   *
   * @param head the head
   * @return list node
   */
  public ListNode middleNode(ListNode head) {
    ListNode lo = head, hi = head;
    while (hi != null && hi.next != null) {
      lo = lo.next;
      hi = hi.next.next;
    }
    return lo;
  }
  /**
   * 回文链表，找中点 & 反转前半部分 & 逐一对比
   *
   * @param head the head
   * @return boolean boolean
   */
  public boolean isPalindrome(ListNode head) {
    ListNode dummy = new ListNode(); // cur 指向当前要反转的节点，dummy 作为头插法的头
    ListNode lo = head, hi = head;
    while (hi != null && hi.next != null) {
      ListNode cur = lo; // 1.暂存
      lo = lo.next;
      hi = hi.next.next; // 步进
      cur.next = dummy.next; // 2.变向
      dummy.next = cur;
    }
    if (hi != null) lo = lo.next; // 此时链表长度为奇数，应该跳过中心节点，否则下方比对 lo 会多一个
    ListNode l1 = dummy.next, l2 = lo; // 分别指向反转后链表的头 & 后半部分链表的头
    while (l1 != null && l2 != null) {
      if (l1.val != l2.val) return false;
      l1 = l1.next;
      l2 = l2.next;
    }
    return true;
  }

  /**
   * 重排链表，找中 & 反转 & 连接
   *
   * @param head the head
   */
  public void reorderList(ListNode head) {
    if (head == null || head.next == null || head.next.next == null) return;
    ListNode lo = middleNode(head);
    ListNode first = lo.next;
    lo.next = null;
    // 第二个链表倒置
    first = reverseList(first);
    // 链表节点依次连接
    while (first != null) {
      ListNode nxt = first.next;
      first.next = head.next;
      head.next = first;
      head = first.next;
      first = nxt;
    }
  }
}

/** 收集合并 & 删除相关 */
class DDoublePointerList extends LList {
  /**
   * 两数相加，模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * <p>I 逆序存储，而 II 正序，因此需要逆序访问
   *
   * @param l1 the l 1
   * @param l2 the l 2
   * @return list node
   */
  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    return addTwoNumbers2(l1, l2);
  }

  private ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
    final int base = 10; // 36 进制
    ListNode dummy = new ListNode();
    int carry = 0;
    ListNode p1 = l1, p2 = l2, cur = dummy;
    while (p1 != null || p2 != null || carry != 0) {
      int n1 = getVal(p1), n2 = getVal(p2);
      int tmp = n1 + n2 + carry;
      carry = tmp / base;
      cur.next = new ListNode(tmp % base);
      p1 = p1 == null ? null : p1.next;
      p2 = p2 == null ? null : p2.next;
      cur = cur.next;
    }
    if (carry == 1) cur.next = new ListNode(1);
    return dummy.next;
  }

  private ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
    Deque<Integer> stack1 = new ArrayDeque<>(), stack2 = new ArrayDeque<>();
    int carry = 0;
    ListNode p1 = l1, p2 = l2, cur = null;
    while (p1 != null) {
      stack1.addLast(p1.val);
      p1 = p1.next;
    }
    while (p2 != null) {
      stack2.addLast(p2.val);
      p2 = p2.next;
    }
    while (!stack1.isEmpty() || !stack2.isEmpty() || carry > 0) {
      carry +=
          (stack1.isEmpty() ? 0 : stack1.removeLast())
              + (stack2.isEmpty() ? 0 : stack2.removeLast());
      ListNode node = new ListNode(carry % 10);
      node.next = cur;
      cur = node;
      carry /= 10;
    }
    return cur;
  }

  /**
   * 合并k个有序链表，分治 up-to-bottom or 大顶堆
   *
   * <p>上方排序链表则为 bottom-to-up
   *
   * @param lists the lists
   * @return the list node
   */
  public ListNode mergeKLists(ListNode[] lists) {
    return (lists.length == 0) ? null : divide(lists, 0, lists.length - 1);
  }

  private ListNode divide(ListNode[] lists, int lo, int hi) {
    if (lo == hi) return lists[lo];
    int mid = lo + (hi - lo) >> 1;
    return mergeTwoLists(divide(lists, lo, mid), divide(lists, mid + 1, hi));
  }

  /**
   * 删除链表的倒数第 N 个结点
   *
   * @param head the head
   * @param n the n
   * @return list node
   */
  public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode lo = dummy, hi = head;
    for (int i = 0; i < n; i++) hi = hi.next;
    while (hi != null) {
      hi = hi.next;
      lo = lo.next;
    }
    lo.next = lo.next.next; // lo.next 即倒数第 n 个
    return dummy.next;
  }

  /**
   * 奇偶链表，先奇后偶 & 变向 & 步进
   *
   * @param head the head
   * @return list node
   */
  public ListNode oddEvenList(ListNode head) {
    if (head == null) return null;
    ListNode evenHead = head.next;
    ListNode odd = head, even = evenHead;
    while (even != null && even.next != null) {
      odd.next = even.next;
      odd = odd.next;
      even.next = odd.next;
      even = even.next;
    }
    odd.next = evenHead;
    return head;
  }

  /**
   * 排序链表，bottom-to-up，即从 2 步长开始分割为 len/2 个 & 合并，最终才把整个分为两端
   *
   * <p>https://leetcode-cn.com/problems/sort-list/solution/148-pai-xu-lian-biao-bottom-to-up-o1-kong-jian-by-/
   *
   * <p>下方合并k个有序链表则 up-to-bottom
   *
   * @param head the head
   * @return list node
   */
  public ListNode sortList(ListNode head) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    int len = 0;
    for (ListNode cur = dummy.next; cur != null; cur = cur.next) len += 1;
    for (int size = 1; size < len; size <<= 1) { // 循环开始切割和合并
      ListNode tail = dummy, cur = dummy.next;
      while (cur != null) {
        ListNode left = cur, right = cut(cur, size); // 链表切掉 size 剩下的返还给 right
        cur = cut(right, size); // 链表切掉 size 剩下的返还给 cur
        tail.next = mergeTwoLists(left, right);
        while (tail.next != null) tail = tail.next; // 保持最尾端
      }
    }
    return dummy.next;
  }
  /**
   * 将链表L切掉前n个节点 并返回后半部分的链表头
   *
   * @param head
   * @param n
   * @return
   */
  private ListNode cut(ListNode head, int n) {
    if (n <= 0) return head;
    ListNode cur = head;
    for (int i = n; i > 0 && cur != null; i--) cur = cur.next;
    if (cur == null) return null;
    ListNode nxt = cur.next;
    cur.next = null;
    return nxt;
  }
  /**
   * 删除排序链表中的重复元素
   *
   * <p>与移动零 & 删除排序数组中的重复项保持一致，符合，即要移动 or 移除的则跳过
   *
   * @param head the head
   * @return list node
   */
  public ListNode deleteDuplicates(ListNode head) {
    return deleteDuplicatesI(head);
  }

  /**
   * 删除排序链表中的重复元素I，保留一个
   *
   * @param head the head
   * @return list node
   */
  private ListNode deleteDuplicatesI(ListNode head) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode cur = dummy.next;
    while (cur != null && cur.next != null) {
      if (cur.val == cur.next.val) cur.next = cur.next.next;
      else cur = cur.next;
    }
    return dummy.next;
  }

  /**
   * 删除排序链表中的重复元素II，毫无保留
   *
   * @param head the head
   * @return list node
   */
  private ListNode deleteDuplicatesII(ListNode head) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode cur = dummy;
    while (cur.next != null && cur.next.next != null) { // diff1
      if (cur.next.val == cur.next.next.val) { // diff2
        int target = cur.next.val;
        while (cur.next != null && cur.next.val == target) cur.next = cur.next.next; // diff3
      } else cur = cur.next;
    }
    return dummy.next;
  }

  /**
   * 分隔链表
   *
   * @param head the head
   * @param x the x
   * @return list node
   */
  public ListNode partition(ListNode head, int x) {
    ListNode ltHead = new ListNode(0), gteHead = new ListNode(0);
    ListNode ltTail = ltHead, gteTail = gteHead;
    ListNode cur = head;
    while (cur != null) {
      // 如果当前节点的值小于x，则把当前节点挂到小链表的后面，变向 & 步进
      if (cur.val < x) {
        ltTail.next = cur;
        ltTail = ltTail.next;
      } else {
        gteTail.next = cur;
        gteTail = gteTail.next;
      }
      cur = cur.next;
    }
    ltTail.next = gteHead.next;
    gteTail.next = null;
    return ltHead.next;
  }
}

/** 收集环形相关 */
class CCycle extends LList {
  /**
   * 环形链表I，判断即可
   *
   * @param head the head
   * @return the boolean
   */
  public boolean hasCycle(ListNode head) {
    boolean cycle = false;
    ListNode lo = head, hi = head;
    while (lo != null && hi != null && hi.next != null) {
      lo = lo.next;
      hi = hi.next.next;
      if (lo == hi) {
        cycle = true;
        break;
      }
    }
    return cycle;
  }

  /**
   * 环形链表II，前半部分与环形链表 I 一致
   *
   * @param head the head
   * @return the list node
   */
  public ListNode detectCycle(ListNode head) {
    boolean hasCycle = false;
    ListNode lo = head, hi = head;
    while (lo != null && hi != null && hi.next != null) {
      lo = lo.next;
      hi = hi.next.next;
      if (lo == hi) {
        hasCycle = true;
        break;
      }
    }
    if (!hasCycle) return null;
    ListNode start = head;
    while (start != lo) {
      start = start.next;
      lo = lo.next;
    }
    return start;
  }
  /**
   * 相交链表 / 两个链表的第一个公共节点，获取两个链表的首个交点，不存在则返空
   *
   * @param headA the head a
   * @param headB the head b
   * @return the intersection node
   */
  public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) return null;
    ListNode l1 = headA, l2 = headB;
    while (l1 != l2) {
      l1 = l1 == null ? headB : l1.next;
      l2 = l2 == null ? headA : l2.next;
    }
    return l1;
  }
}

/** 收集反转相关 */
class RReverseList extends LList {
  /**
   * k个一组反转链表，三步曲，暂存 & 变向 & 步进
   *
   * @param head the head
   * @param k the k
   * @return the list node
   */
  public ListNode reverseKGroup(ListNode head, int k) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode tail = dummy, cur = dummy;
    while (cur.next != null) {
      for (int i = 0; i < k && cur != null; i++) cur = cur.next;
      if (cur == null) break;
      ListNode first = tail.next, nxt = cur.next; // 1.暂存
      cur.next = null; // 2.此时 cur 是 last，断开
      tail.next = reverseList(first); // 3.变向两次
      first.next = nxt;
      tail = first; // 4.步进
      cur = first;
    }
    return dummy.next;
  }

  /**
   * 两两交换链表中的节点，暂存 & 变向 & 步进，亦可复用 reverseKGroup
   *
   * @param head the head
   * @return list node
   */
  public ListNode swapPairs(ListNode head) {
    // return reverseKGroup(head, 2);
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode cur = dummy;
    while (cur.next != null && cur.next.next != null) {
      ListNode first = cur.next, second = cur.next.next; // 暂存
      cur.next = second; // 步进
      first.next = second.next;
      second.next = first;
      cur = first; // 步进
    }
    return dummy.next;
  }

  /**
   * 反转链表II，尾插，三步曲，暂存 & 变向
   *
   * @param head the head
   * @param left the left
   * @param right the right
   * @return the list node
   */
  public ListNode reverseBetween(ListNode head, int left, int right) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    ListNode tail = dummy;
    for (int step = 0; step < left - 1; step++) tail = tail.next;
    ListNode first = tail.next; // tail first cur nxt
    ListNode cur = first.next, nxt = null;
    for (int i = 0; i < right - left; i++) { // 尾插
      nxt = cur.next; // 1.暂存
      cur.next = tail.next; // 2.变向三次
      tail.next = cur;
      first.next = nxt;
      cur = nxt; // 3.步进
    }
    return dummy.next;
  }

  /**
   * 旋转链表，闭环后断开
   *
   * @param head the head
   * @param k the k
   * @return list node
   */
  public ListNode rotateRight(ListNode head, int k) {
    if (head == null || head.next == null || k == 0) return head;
    int len = 1;
    ListNode cur = head;
    while (cur.next != null) {
      cur = cur.next;
      len += 1;
    }
    int add = len - k % len;
    if (add == len) return head;
    cur.next = head; // 闭环
    while (add-- > 0) cur = cur.next; // 断点
    ListNode res = cur.next;
    cur.next = null; // 断开
    return res;
  }
}
