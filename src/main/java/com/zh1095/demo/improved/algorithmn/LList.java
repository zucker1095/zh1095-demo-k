package com.zh1095.demo.improved.algorithmn;

import java.util.*;

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
  /**
   * 反转链表，三步曲，暂存 1/2 & 变向 1/2/3 & 步进 1/2 次，pre cur nxt
   *
   * @param head the head
   * @return the list node
   */
  public ListNode reverseList(ListNode head) {
    // 递归
    //    if (head == null || head.next == null) return head;
    //    // 后半部分反转的头，即反转前 head 链表的尾
    //    ListNode newHead = reverseList(head.next);
    //    head.next.next = head;
    //    head.next = null;
    //    return newHead;
    // 迭代
    ListNode pre = null, cur = head, nxt;
    while (cur != null) {
      // 1.暂存
      nxt = cur.next;
      // 2.变向
      cur.next = pre;
      // 3.步进
      pre = cur;
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
    ListNode dummy = new ListNode(0);
    ListNode cur = dummy, l1 = list1, l2 = list2;
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
   * 复制带随机指针的链表，即深拷贝，三次遍历
   *
   * <p>参考
   * https://leetcode-cn.com/problems/copy-list-with-random-pointer/solution/liang-chong-shi-xian-tu-jie-138-fu-zhi-dai-sui-ji-/
   *
   * @param head
   * @return
   */
  public Node copyRandomList(Node head) {
    if (head == null) return null;
    Node cur = head;
    // 1.在每个原节点后面创建一个新节点
    while (cur != null) {
      Node newNode = new Node(cur.val);
      newNode.next = cur.next;
      cur.next = newNode;
      cur = newNode.next;
    }
    // 2.逐一设置新节点的随机节点
    cur = head;
    while (cur != null) {
      if (cur.random != null) {
        cur.next.random = cur.random.next;
      }
      cur = cur.next.next;
    }
    // 3.分离两个链表
    Node dummy = new Node(-1);
    Node lo = dummy, hi = head;
    while (hi != null) {
      lo.next = hi.next;
      lo = lo.next;
      hi.next = lo.next;
      hi = hi.next;
    }
    return dummy.next;
  }

  /**
   * 链表中的下一个更大节点，单调栈
   *
   * <p>TODO
   *
   * @param head
   * @return
   */
  public int[] nextLargerNodes(ListNode head) {
    List<Integer> res = new ArrayList<>();
    Deque<Integer> monotonousStack = new ArrayDeque<>();
    ListNode cur = head;
    while (cur != null) {
      while (!monotonousStack.isEmpty() && cur.val > res.get(monotonousStack.peekLast())) {
        res.set(monotonousStack.pollLast(), cur.val);
      }
      monotonousStack.offerLast(res.size());
      res.add(cur.val);
      cur = cur.next;
    }
    for (int i : monotonousStack) {
      res.set(i, 0);
    }
    return res.stream().mapToInt(i -> i).toArray();
  }

  private class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
      this.val = val;
      this.next = null;
      this.random = null;
    }
  }
}

/** 收集反转相关 */
class ReverseList extends LList {
  /**
   * k个一组反转链表，tail [first ... cur] nxt
   *
   * <p>变向顺序 cur tail first，只需专注其后驱即可，下方反转区间同理
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
    ListNode tail = dummy, first = null, cur = dummy, nxt = null;
    while (cur.next != null) {
      // 扩展1，cur.next!=null 并统计 i 是否为 k 因为可能刚好等于 k，且不需判空当前结点
      //      for (int i = 0; i < k && cur.next != null; i++) {
      //        cur = cur.next;
      //      }
      for (int i = 0; i < k && cur != null; i++) {
        cur = cur.next;
      }
      // 此时 cur 为区间尾
      if (cur == null) break;
      first = tail.next;
      nxt = cur.next;
      cur.next = null;
      tail.next = reverseList(first);
      first.next = nxt;
      // 步进
      tail = first;
      cur = first;
    }
    return dummy.next;
  }

  /**
   * 反转链表II，区间，tail [first cur...] nxt
   *
   * <p>三步曲，暂存 & 变向三次 & 步进，顺序同上 cur tail first
   *
   * @param head the head
   * @param left the left
   * @param right the right
   * @return the list node
   */
  public ListNode reverseBetween(ListNode head, int left, int right) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    ListNode tail = dummy;
    for (int step = 0; step < left - 1; step++) {
      tail = tail.next;
    }
    ListNode first = tail.next, cur = first.next, nxt = cur.next;
    for (int i = 0; i < right - left; i++) {
      nxt = cur.next;
      cur.next = tail.next;
      tail.next = cur;
      first.next = nxt;
      cur = nxt;
    }
    return dummy.next;
  }

  /**
   * 回文链表，找中点，同时反转前半部分 & 逐一比对两条链表
   *
   * @param head the head
   * @return boolean boolean
   */
  public boolean isPalindrome(ListNode head) {
    ListNode tail = new ListNode(), lo = head, hi = head;
    while (hi != null && hi.next != null) {
      ListNode cur = lo;
      lo = lo.next;
      hi = hi.next.next;
      // 两次变向，头插
      cur.next = tail.next;
      tail.next = cur;
    }
    // 此时链表长度为奇数，应该跳过中心节点，否则下方比对 lo 会多一位
    if (hi != null) lo = lo.next;
    ListNode l1 = tail.next, l2 = lo;
    while (l1 != null && l2 != null) {
      if (l1.val != l2.val) return false;
      l1 = l1.next;
      l2 = l2.next;
    }
    return true;
  }

  /**
   * 旋转链表，特判三空 & 获取链尾与总长 & 闭环 & 断开
   *
   * @param head the head
   * @param k the k
   * @return list node
   */
  public ListNode rotateRight(ListNode head, int k) {
    if (head == null || head.next == null || k == 0) {
      return head;
    }
    int len = 1;
    ListNode cur = head;
    while (cur.next != null) {
      cur = cur.next;
      len += 1;
    }
    int count = len - k % len;
    if (count == len) return head;
    cur.next = head;
    for (int i = count; i > 0; i--) {
      cur = cur.next;
    }
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

/** 收集合并，重排相关 */
class MergeList extends LList {
  /**
   * 两数相加，本质即外排，考虑进位即可
   *
   * <p>模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
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

  // 123 & 45 -> 573
  private ListNode addTwoNumbers1(ListNode l1, ListNode l2) {
    final int base = 10; // 36 进制
    ListNode dummy = new ListNode();
    int carry = 0;
    ListNode p1 = l1, p2 = l2, cur = dummy;
    while (p1 != null || p2 != null || carry != 0) {
      int n1 = p1 == null ? 0 : p1.val, n2 = p2 == null ? 0 : p2.val;
      int tmp = n1 + n2 + carry;
      carry = tmp / base;
      cur.next = new ListNode(tmp % base);
      p1 = p1 == null ? null : p1.next;
      p2 = p2 == null ? null : p2.next;
      cur = cur.next;
    }
    return dummy.next;
  }

  // 123 & 45 -> 168
  // 分别反转两个链表 & 正序计算 & 反转整个链表
  // 允许空间，则分别遍历建栈，注意需要逆序尾插
  private ListNode addTwoNumbers2(ListNode l1, ListNode l2) {
    //    ListNode p1 = reverseList(l1), p2 = reverseList(l2);
    //    return reverseList(addTwoNumbers1(p1, p2));
    Deque<Integer> st1 = new ArrayDeque<>(), st2 = new ArrayDeque<>();
    while (l1 != null) {
      st1.push(l1.val);
      l1 = l1.next;
    }
    while (l2 != null) {
      st2.push(l2.val);
      l2 = l2.next;
    }
    ListNode head = null;
    int carry = 0;
    while (!st1.isEmpty() || !st2.isEmpty() || carry > 0) {
      int n1 = st1.isEmpty() ? 0 : st1.pop(), n2 = st2.isEmpty() ? 0 : st2.pop();
      int tmp = n1 + n2 + carry;
      carry = tmp / 10;
      ListNode cur = new ListNode(tmp % 10);
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
              if (n1.val < n2.val) return -1;
              else if (n1.val == n2.val) return 0;
              else return 1;
            });
    for (ListNode node : lists) {
      if (node == null) continue;
      pq.add(node);
    }
    ListNode dummy = new ListNode(0), cur = dummy;
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

  /**
   * 奇偶链表，如 1234 至 1324，暂存偶头 & 奇偶 & 变向步进
   *
   * <p>扩展1，排序一个奇数位升序而偶数位降序的链表，O(n) & O(1)，参下「重排奇偶链表」
   *
   * @param head the head
   * @return list node
   */
  public ListNode oddEvenList(ListNode head) {
    if (head == null) {
      return null;
    }
    ListNode odd = head, even = head.next;
    ListNode evenHead = head.next;
    while (even != null && even.next != null) {
      odd.next = even.next;
      odd = odd.next;
      even.next = odd.next;
      even = even.next;
    }
    // 奇尾连偶头
    odd.next = evenHead;
    return head;
  }

  /**
   * 重排奇偶链表，分别取出奇偶链表，奇数位链表需断尾 & 反转偶数位链表 & 合并
   *
   * <p>参考 https://mp.weixin.qq.com/s/0WVa2wIAeG0nYnVndZiEXQ
   *
   * @param head
   * @return
   */
  public ListNode sortOddEvenList(ListNode head) {
    ListNode oddHead = head, evenHead = _oddEvenList(head);
    evenHead = reverseList(evenHead);
    return mergeTwoLists(oddHead, evenHead);
  }

  // 代码与上方「奇偶链表」一致，但返回偶头
  private ListNode _oddEvenList(ListNode head) {
    if (head == null) {
      return null;
    }
    ListNode odd = head, even = head.next;
    ListNode evenHead = head.next;
    while (even != null && even.next != null) {
      odd.next = even.next;
      odd = odd.next;
      even.next = odd.next;
      even = even.next;
    }
    odd.next = null;
    return evenHead;
  }

  /**
   * 排序链表，建议掌握递归 up-to-bottom 即可
   *
   * <p>扩展1，去重，参下 annotate
   *
   * @param head the head
   * @return list node
   */
  public ListNode sortList(ListNode head) {
    return sortList1(head);
  }

  // 找中点 & 分割 & 分别排序 & 合并
  private ListNode sortList1(ListNode head) {
    if (head == null || head.next == null) return head;
    ListNode lo = head, hi = head.next;
    while (hi != null && hi.next != null) {
      lo = lo.next;
      hi = hi.next.next;
    }
    ListNode tmp = lo.next;
    lo.next = null;
    ListNode left = sortList1(head), right = sortList1(tmp);
    return mergeTwoLists(left, right);
  }

  private ListNode sortList2(ListNode head) {
    ListNode dummy = new ListNode();
    dummy.next = head;
    int len = 0;
    ListNode node = dummy.next;
    while (node != null) {
      len += 1;
      node = node.next;
    }
    // 循环开始切割和合并
    for (int size = 1; size < len; size /= 2) {
      ListNode cur = dummy.next, tail = dummy;
      while (cur != null) {
        // 链表切掉 size 剩下的返还给 right
        ListNode left = cur, right = cut(cur, size);
        // 链表切掉 size 剩下的返还给 cur
        cur = cut(right, size);
        tail.next = mergeTwoLists(left, right);
        // 保持最尾端
        while (tail.next != null) {
          tail = tail.next;
        }
      }
    }
    return dummy.next;
  }

  // 将链表切掉前 n 个节点，并返回后半部分的链表头
  private ListNode cut(ListNode head, int n) {
    if (n <= 0) return head;
    ListNode cur = head;
    // 往前走 n-1 步
    while (n > 1 && cur != null) {
      cur = cur.next;
      n -= 1;
    }
    if (cur == null) return null;
    ListNode nxt = cur.next;
    cur.next = null;
    return nxt;
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
    // 至少需要有两个点
    if (head == null || head.next == null || head.next.next == null) {
      return;
    }
    ListNode p1 = head, p1Tail = middleNode(head), p2 = p1Tail.next, p2Nxt;
    p1Tail.next = null;
    p2 = reverseList(p2);
    // 合并两个链表即可
    while (p2 != null) {
      p2Nxt = p2.next;
      p2.next = p1.next;
      p1.next = p2;
      p1 = p2.next;
      p2 = p2Nxt;
    }
  }

  /**
   * 分隔链表，使得所有小于 x 的结点都出现在大于或等于 x 的结点之前
   *
   * <p>三链表，类似双路快排，尾插
   *
   * @param head the head
   * @param x the x
   * @return list node
   */
  public ListNode partition(ListNode head, int x) {
    // head
    ListNode ltDummy = new ListNode(), gteDummy = new ListNode();
    // tail
    ListNode ltCur = ltDummy, gteCur = gteDummy, cur = head;
    while (cur != null) {
      // 如果当前节点的值小于x，则把当前节点挂到小链表的后面
      if (cur.val < x) {
        ltCur.next = cur;
        ltCur = ltCur.next;
      } else {
        gteCur.next = cur;
        gteCur = gteCur.next;
      }
      cur = cur.next;
    }
    ltCur.next = gteDummy.next;
    gteCur.next = null;
    return ltDummy.next;
  }
}

/** 收集环形，搜索相关 */
class CycleList extends LList {
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
   * <p>扩展1，求环路长度，遍历回到入口即可
   *
   * <p>扩展2，打印环首尾，后者指 tail.next=entry
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
    ListNode finder = head;
    while (lo != finder) {
      lo = lo.next;
      finder = finder.next;
    }
    return lo;
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

/** 删除相关 */
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
   * 删除排序链表中的重复元素I，保留一个，即去重
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
    ListNode pre = dummy, cur = dummy.next;
    while (cur != null && cur.next != null) {
      if (pre.next.val == cur.next.val) {
        int target = pre.next.val;
        while (cur.next != null && target == cur.next.val) {
          cur = cur.next;
        }
        pre.next = cur.next;
      } else {
        pre = cur;
      }
      cur = cur.next;
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
    Map<Integer, ListNode> preSumByNode = new HashMap<>();
    // 建立前缀和，覆盖取最终出现的结点
    int preSum = 0;
    while (cur != null) {
      preSum += cur.val;
      preSumByNode.put(preSum, cur);
      cur = cur.next;
    }
    // 若当前节点处 sum 在下一处出现，则表明两结点之间所有节点和为零，因此删除区间所有节点
    preSum = 0;
    cur = dummy;
    while (cur != null) {
      preSum += cur.val;
      cur.next = preSumByNode.get(preSum).next;
      cur = cur.next;
    }
    return dummy.next;
  }
}
