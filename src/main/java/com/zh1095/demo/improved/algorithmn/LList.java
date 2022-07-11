package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集所有链表相关，建议直接记代码行数，因为链表的题型基本没有冗余的步骤
 *
 * <p>头插需要引入哑节点
 *
 * @author cenghui
 */
public class LList {
  /**
   * 反转链表，三步曲，暂存 1/2 & 变向 1/2/3 & 步进 1/2 次，pre cur nxt
   *
   * @param head the head
   * @return the list node
   */
  public ListNode reverseList(ListNode head) {
    // recursion
    //    if (head == null || head.next == null) return head;
    //    // 后半部分反转的头，即反转前 head 链表的尾
    //    ListNode newHead = reverseList(head.next);
    //    head.next.next = head;
    //    head.next = null;
    //    return newHead;
    // iteration
    ListNode pre = null, cur = head;
    while (cur != null) {
      ListNode nxt = cur.next; // 暂存
      cur.next = pre; // 变向
      pre = cur; // 步进
      cur = nxt;
    }
    return pre;
  }

  /**
   * 合并两个有序链表，正向，参考合并两个有序数组
   *
   * <p>模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * <p>扩展1，去重但不能合并，按顺序打印，则每次比对 res[res.length-1] 与二者更大即可
   *
   * @param list1 the list 1
   * @param list2 the list 2
   * @return the list node
   */
  public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    ListNode dummy = new ListNode(), cur = dummy, l1 = list1, l2 = list2;
    while (l1 != null || l2 != null) {
      int n1 = l1 == null ? Integer.MAX_VALUE : l1.val,
          n2 = l2 == null ? Integer.MAX_VALUE : l2.val;
      if (n1 < n2) {
        cur.next = l1;
        l1 = l1.next;
      } else {
        cur.next = l2;
        l2 = l2.next;
      }
      cur = cur.next;
    }
    return dummy.next;
  }

  /**
   * 复制带随机指针的链表，三次遍历，一次需要引入 hash
   *
   * <p>参考
   * https://leetcode-cn.com/problems/copy-list-with-random-pointer/solution/liang-chong-shi-xian-tu-jie-138-fu-zhi-dai-sui-ji-/
   *
   * @param head
   * @return
   */
  public Node copyRandomList(Node head) {
    // 1.在每个原节点后面创建一个新节点
    Node cur = head;
    while (cur != null) {
      Node newNode = new Node(cur.val);
      newNode.next = cur.next;
      cur.next = newNode;
      cur = newNode.next;
    }
    // 2.逐一设置新节点的随机节点
    cur = head;
    while (cur != null && cur.next != null) {
      // 保证 DAG
      if (cur.random != null) cur.next.random = cur.random.next;
      cur = cur.next.next;
    }
    // 3.分离
    Node dummy = new Node(-1), lo = dummy, hi = head;
    while (hi != null) {
      lo.next = hi.next;
      lo = lo.next;
      hi.next = lo.next;
      hi = hi.next;
    }
    return dummy.next;
  }

  /**
   * 链表的中间结点，快慢指针
   *
   * @param head the head
   * @return list node
   */
  public ListNode middleNode(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode lo = head, hi = head.next.next;
    while (hi != null && hi.next != null) {
      lo = lo.next;
      hi = hi.next.next;
    }
    return lo;
  }

  /**
   * 链表中的下一个更大节点，单调栈
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/next-greater-node-in-linked-list/solution/javati-jie-dan-diao-zhan-fa-by-maugahm-4-gl74/
   *
   * @param head
   * @return
   */
  public int[] nextLargerNodes(ListNode head) {
    List<Integer> nodes = new ArrayList<>();
    Deque<Integer> ms = new ArrayDeque<>();
    ListNode cur = head;
    while (cur != null) {
      while (!ms.isEmpty() && cur.val > nodes.get(ms.peekLast())) {
        nodes.set(ms.pollLast(), cur.val);
      }
      ms.offerLast(nodes.size());
      nodes.add(cur.val);
      cur = cur.next;
    }
    for (int i : ms) nodes.set(i, 0);
    return nodes.stream().mapToInt(i -> i).toArray();
  }

  private class Node {
    int val;
    Node next, random;

    public Node(int val) {
      this.val = val;
    }
  }
}

/** 收集反转相关 */
class ReverseList extends LList {
  /**
   * k个一组反转链表，pre [start...cur...] nxt
   *
   * <p>只需专注后驱，下方反转区间同理
   *
   * <p>参考
   * https://leetcode-cn.com/problems/reverse-nodes-in-k-group/solution/tu-jie-kge-yi-zu-fan-zhuan-lian-biao-by-user7208t/
   *
   * <p>扩展1，不足 k 个也反转，参下 annotate
   *
   * <p>扩展2，从尾部开始计数，如 12345 & k=2 为 13254，先遍历一趟获取长度
   *
   * @param head the head
   * @param k the k
   * @return the list node
   */
  public ListNode reverseKGroup(ListNode head, int k) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode pre = dummy, cur = dummy;
    while (cur.next != null) {
      // cur.next != null 并统计 i 是否为 k 因为可能刚好等于 k，且不需判空当前结点
      //      for (int i = 0; i < k && cur.next != null; i++) {
      //        cur = cur.next;
      //      }
      for (int i = 0; i < k && cur != null; i++) {
        cur = cur.next;
      }
      if (cur == null) break;
      // 此时 cur 为区间尾，暂存
      ListNode start = pre.next, nxt = cur.next;
      // 变向
      cur.next = null;
      pre.next = reverseList(start);
      start.next = nxt;
      // 步进
      pre = start;
      cur = start;
    }
    return dummy.next;
  }

  /**
   * 反转链表II，区间，pre [start...cur...] nxt
   *
   * <p>三步曲，暂存 & 变向三次 & 步进，顺序同上 cur pre start
   *
   * @param head the head
   * @param left the left
   * @param right the right
   * @return the list node
   */
  public ListNode reverseBetween(ListNode head, int left, int right) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode pre = dummy;
    for (int step = 0; step < left - 1; step++) {
      pre = pre.next;
    }
    // 题设保证区间合法，因此 pre.next.next 非空
    ListNode start = pre.next, cur = pre.next.next;
    for (int i = 0; i < right - left; i++) {
      ListNode nxt = cur.next;
      cur.next = pre.next;
      pre.next = cur;
      start.next = nxt;
      cur = nxt;
    }
    return dummy.next;
  }

  /**
   * 回文链表，找中点，同时反转前半部分 & 逐一比对两条链表，相当于中间扩散
   *
   * @param head the head
   * @return boolean boolean
   */
  public boolean isPalindrome(ListNode head) {
    ListNode dummy = new ListNode(), lo = head, hi = head;
    while (hi != null && hi.next != null) {
      ListNode cur = lo;
      // 头插
      cur.next = dummy.next;
      dummy.next = cur;
      // 步进
      lo = lo.next;
      hi = hi.next.next;
    }
    // 长度为奇数应跳过中点，否则下方比对 lo 会多一位
    if (hi != null) lo = lo.next;
    ListNode l1 = dummy.next, l2 = lo;
    while (l1 != null && l2 != null) {
      if (l1.val != l2.val) return false;
      l1 = l1.next;
      l2 = l2.next;
    }
    return true;
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
    ListNode cur = head;
    // 1.count length
    int len = 1;
    while (cur.next != null) {
      cur = cur.next;
      len += 1;
    }
    // 2.find the next count node
    int count = len - k % len;
    if (count == len) return head;
    cur.next = head;
    for (int i = count; i > 0; i--) {
      cur = cur.next;
    }
    // 3.merge and separate
    ListNode newHead = cur.next;
    cur.next = null;
    return newHead;
  }

  /**
   * 两两交换链表中的节点，复用 reverseKGroup 即可
   *
   * @param head the head
   * @return list node
   */
  public ListNode swapPairs(ListNode head) {
    return reverseKGroup(head, 2);
  }
}

/** 收集合并相关 */
class MergeList extends LList {
  /**
   * 两数相加，本质即外排，考虑进位即可
   *
   * <p>模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * <p>I 逆序 II 正序
   *
   * @param l1 the l 1
   * @param l2 the l 2
   * @return list node
   */
  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    return addTwoNumbers2(l1, l2);
  }

  // 123 & 45 -> 573
  private ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
    final int base = 10; // 36 进制
    ListNode dummy = new ListNode(), cur = dummy, p1 = l1, p2 = l2;
    int carry = 0; // 还要加上一个高位
    while (p1 != null || p2 != null || carry != 0) {
      int n1 = p1 == null ? 0 : p1.val, n2 = p2 == null ? 0 : p2.val, tmp = n1 + n2 + carry;
      cur.next = new ListNode(tmp % base);
      cur = cur.next;
      carry = tmp / base;
      p1 = p1 == null ? null : p1.next;
      p2 = p2 == null ? null : p2.next;
    }
    return dummy.next;
  }

  // 123 & 45 -> 168
  // 分别反转 & 正序计算 & 反转整个链表
  // 允许空间，则分别遍历建栈，注意需要逆序尾插，即 cur->head
  private ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
    //    ListNode p1 = reverseList(l1), p2 = reverseList(l2);
    //    return reverseList(addTwoNumbers1(p1, p2));
    final int base = 10; // 36 进制
    Deque<Integer> st1 = new ArrayDeque<>(), st2 = new ArrayDeque<>();
    while (l1 != null) {
      st1.offerLast(l1.val);
      l1 = l1.next;
    }
    while (l2 != null) {
      st2.offerLast(l2.val);
      l2 = l2.next;
    }
    ListNode head = null;
    int carry = 0;
    while (!st1.isEmpty() || !st2.isEmpty() || carry > 0) {
      int n1 = st1.isEmpty() ? 0 : st1.pollLast(),
          n2 = st2.isEmpty() ? 0 : st2.pollLast(),
          tmp = n1 + n2 + carry;
      carry = tmp / base;
      ListNode cur = new ListNode(tmp % base);
      cur.next = head;
      head = cur;
    }
    return head;
  }

  /**
   * 合并k个有序链表，大顶堆 / 分治 up-to-bottom
   *
   * <p>下方「排序链表」则为 bottom-to-up
   *
   * @param lists the lists
   * @return the list node
   */
  public ListNode mergeKLists(ListNode[] lists) {
    if (lists.length < 1) return null;
    Queue<ListNode> pq =
        new PriorityQueue<>(
            lists.length,
            (ListNode n1, ListNode n2) -> {
              return n1.val - n2.val;
            });
    for (ListNode head : lists) {
      pq.offer(head);
    }
    ListNode dummy = new ListNode(), cur = dummy;
    while (!pq.isEmpty()) {
      cur.next = pq.poll();
      cur = cur.next;
      if (cur.next != null) pq.offer(cur.next);
    }
    return dummy.next;
    //    return (lists.length == 0) ? null : divide(lists, 0, lists.length - 1);
  }

  private ListNode divide(ListNode[] lists, int lo, int hi) {
    if (lo == hi) return lists[lo];
    int mid = lo + (hi - lo) / 2;
    return mergeTwoLists(divide(lists, lo, mid), divide(lists, mid + 1, hi));
  }
}

/** 重排链表 */
class ReorderList extends LList {
  /**
   * 奇偶链表，如将 1234 调整为 1324，分离为两条链即可。
   *
   * <p>扩展1，排序一个奇数位升序而偶数位降序的链表，O(n) & O(1)，参下「重排奇偶链表」
   *
   * @param head the head
   * @return list node
   */
  public ListNode oddEvenList(ListNode head) {
    if (head == null) return null;
    ListNode evenHead = head.next, oddTail = getOddTail(head);
    oddTail.next = evenHead;
    return head;
  }

  /**
   * 重排奇偶链表，奇数位置升序，偶数位置反之，升序排列整个链表
   *
   * <p>参考 https://mp.weixin.qq.com/s/0WVa2wIAeG0nYnVndZiEXQ
   *
   * @param head
   * @return
   */
  public ListNode sortOddEvenList(ListNode head) {
    if (head == null) return null;
    ListNode evenHead = head.next, oddTail = getOddTail(head);
    oddTail.next = null;
    return mergeTwoLists(head, reverseList(evenHead));
  }

  private ListNode getOddTail(ListNode head) {
    if (head == null) return null;
    ListNode odd = head, even = head.next;
    while (even != null && even.next != null) {
      odd.next = even.next;
      odd = odd.next;
      even.next = odd.next;
      even = even.next;
    }
    return odd;
  }

  /**
   * 排序链表，建议掌握递归 up-to-bottom 即可，找中点 & 分割 & 分别排序 & 合并
   *
   * <p>bottom-to-up 参考
   * https://leetcode.cn/problems/sort-list/solution/pai-xu-lian-biao-di-gui-die-dai-xiang-jie-by-cherr/
   *
   * <p>扩展1，去重，参下 annotate
   *
   * @param head the head
   * @return list node
   */
  public ListNode sortList(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode mid = middleNode(head), head2 = mid.next;
    mid.next = null;
    return mergeTwoLists(sortList(head), sortList(head2));
    //    ListNode dummy = new ListNode();
    //    dummy.next = head;
    //    // 1.count length
    //    int len = 0;
    //    while (head != null) {
    //      len += 1;
    //      head = head.next;
    //    }
    //    // 2.依次将链表分成1块，2块，4块...
    //    for (int step = 1; step < len; step *= 2) {
    //      // 每次变换步长，pre 和 cur 都初始化在链表头
    //      ListNode pre = dummy, cur = dummy.next;
    //      while (cur != null) {
    //        ListNode h1 = cur, h2 = split(h1, step);
    //        cur = split(h2, step);
    //        pre.next = mergeTwoLists(h1, h2);
    //        // 3.pre 步进到排序好的部分的末尾
    //        while (pre.next != null) pre = pre.next;
    //      }
    //    }
    //    return dummy.next;
  }

  private ListNode split(ListNode head, int step) {
    if (head == null) return null;
    ListNode l1 = head;
    for (int i = 1; i < step && l1.next != null; i++) {
      l1 = l1.next;
    }
    ListNode l2 = l1.next;
    l1.next = null;
    return l2;
  }

  // 链表快排，时间复杂度炸裂
  private ListNode quickSort(ListNode head, ListNode end) {
    if (head == end || head.next == end) {
      return head;
    }
    // 分别为头插与尾插，变向 & 步进
    ListNode ltHead = head, gteTail = head;
    ListNode cur = head.next, nxt;
    while (cur != end) {
      nxt = cur.next;
      if (cur.val < head.val) {
        cur.next = ltHead;
        ltHead = cur;
      } else {
        gteTail.next = cur;
        gteTail = cur;
      }
      cur = nxt;
    }
    gteTail.next = end;
    // 顺序要求先 lt 再 gte
    ListNode node = quickSort(ltHead, head);
    head.next = quickSort(head.next, end);
    return node;
  }

  /**
   * 重排链表，类似奇偶链表，将 1,2,3...n-1,n 排序为 1,n,2,n-1,3...n/2
   *
   * <p>找中 & 反转 & 连接
   *
   * @param head the head
   */
  public void reorderList(ListNode head) {
    // 偶数个节点返回前一个
    ListNode l1 = head, mid = middleNode(head), l2 = mid.next;
    mid.next = null;
    l2 = reverseList(l2); // 翻转
    while (l1 != null && l2 != null) { // 逐个尾插
      ListNode l1Nxt = l1.next, l2Nxt = l2.next;
      l1.next = l2;
      l1 = l1Nxt;
      l2.next = l1;
      l2 = l2Nxt;
    }
  }

  /**
   * 分隔链表，使得所有小于 x 的结点都出现在大于或等于 x 的结点之前
   *
   * <p>两条链表，类似双路快排，尾插
   *
   * @param head the head
   * @param x the x
   * @return list node
   */
  public ListNode partition(ListNode head, int x) {
    ListNode ltDummy = new ListNode(), gteDummy = new ListNode();
    ListNode ltHead = ltDummy, gteHead = gteDummy;
    for (ListNode cur = head; cur != null; cur = cur.next) {
      if (cur.val < x) {
        ltHead.next = cur;
        ltHead = ltHead.next;
      } else {
        gteHead.next = cur;
        gteHead = gteHead.next;
      }
    }
    // merge
    ltHead.next = gteDummy.next;
    gteHead.next = null;
    return ltDummy.next;
  }
}

/** 收集环形，搜索相关 */
class Cyclic extends LList {
  /**
   * 环形链表I，判断即可
   *
   * <p>扩展1，打印首尾
   *
   * <p>扩展2，走三步能否判断有环
   *
   * @param head the head
   * @return the boolean
   */
  public boolean hasCycle(ListNode head) {
    return cyclic(head) != null;
  }

  /**
   * 环形链表II
   *
   * <p>扩展1，求环路长度，遍历回到入口即可
   *
   * <p>扩展2，打印环首尾，后者指 pre.next=entry
   *
   * @param head the head
   * @return the list node
   */
  public ListNode detectCycle(ListNode head) {
    ListNode meet = cyclic(head);
    if (meet == null) return null;
    ListNode finder = head;
    while (meet != finder) {
      meet = meet.next;
      finder = finder.next;
    }
    return meet;
  }

  private ListNode cyclic(ListNode head) {
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
    return cycle ? lo : null;
  }

  /**
   * 相交链表/两个链表的第一个公共节点，获取两个链表的首个交点，不存在则返空
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

/** 删除相关，数组遇到目标则 skip 而链表是变向 */
class DeleteList extends LList {
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
    for (int i = 0; i < n; i++) {
      hi = hi.next;
    }
    while (hi != null) {
      hi = hi.next;
      lo = lo.next;
    }
    lo.next = lo.next.next;
    return dummy.next;
  }

  /**
   * 删除排序链表中的重复元素
   *
   * <p>与移动零 & 删除排序数组中的重复项保持一致，符合，即要移动 or 移除的则跳过
   *
   * <p>TODO 类似滑窗，后期考虑同步后者的模板
   *
   * @param head the head
   * @return list node
   */
  public ListNode deleteDuplicates(ListNode head) {
    return deleteDuplicatesI(head);
  }

  /**
   * 删除排序链表中的重复元素I，保留一个，即链表去重
   *
   * @param head the head
   * @return list node
   */
  private ListNode deleteDuplicatesI(ListNode head) {
    ListNode dummy = new ListNode(), cur = head;
    dummy.next = head;
    while (cur != null && cur.next != null) {
      if (cur.val != cur.next.val) cur = cur.next;
      else cur.next = cur.next.next; // 进入迭代 next 非空
    }
    return dummy.next;
  }

  /**
   * 删除排序链表中的重复元素II，毫无保留，因此需要保留前一个结点的指针
   *
   * @param head the head
   * @return list node
   */
  private ListNode deleteDuplicatesII(ListNode head) {
    ListNode dummy = new ListNode(), pre = dummy;
    dummy.next = head;
    while (pre.next != null && pre.next.next != null) {
      if (pre.next.val != pre.next.next.val) {
        pre = pre.next;
        continue;
      }
      int pivot = pre.next.val; // 进入迭代非空
      while (pre.next != null && pre.next.val == pivot) {
        pre.next = pre.next.next;
      }
    }
    return dummy.next;
  }

  /**
   * 从链表中删去总和值为零的连续节点，前缀和
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/solution/java-hashmap-liang-ci-bian-li-ji-ke-by-shane-34/
   *
   * @param head
   * @return
   */
  public ListNode removeZeroSumSublists(ListNode head) {
    ListNode dummy = new ListNode(), cur = dummy;
    dummy.next = head;
    Map<Integer, ListNode> presumByLastNode = new HashMap<>();
    // 建立前缀和，覆盖取最终出现的结点
    int presum = 0;
    while (cur != null) {
      presum += cur.val;
      presumByLastNode.put(presum, cur);
      cur = cur.next;
    }
    // 若当前节点处 sum 在下一处出现，则表明两结点之间所有节点和为零，因此删除区间所有节点
    presum = 0;
    cur = dummy;
    while (cur != null) {
      presum += cur.val;
      cur.next = presumByLastNode.get(presum).next;
      cur = cur.next;
    }
    return dummy.next;
  }
}
