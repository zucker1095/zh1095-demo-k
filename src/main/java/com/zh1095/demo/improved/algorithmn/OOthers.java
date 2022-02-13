package com.zh1095.demo.improved.algorithmn;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;

/**
 * 收集非五大基本类型的
 *
 * @author cenghui
 */
public class OOthers {
  /**
   * 和为k的子数组，前缀和 value：key 对应的前缀和的个数
   *
   * @param nums the nums
   * @param k the k
   * @return int int
   */
  public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> freqByPreSum =
        new HashMap() {
          {
            put(0, 1); // 对于下标为 0 的元素，前缀和为 0，个数为 1
          }
        };
    int preSum = 0, count = 0;
    for (int num : nums) {
      preSum += num;
      // 先获得前缀和为 preSum - k 的个数，加到计数变量里
      if (freqByPreSum.containsKey(preSum - k)) count += freqByPreSum.get(preSum - k);
      // 然后维护 preSumFreq 的定义
      freqByPreSum.put(preSum, freqByPreSum.getOrDefault(preSum, 0) + 1);
    }
    return count;
  }

  /**
   * 划分字母区间
   *
   * @param s the s
   * @return list
   */
  //  public List<Integer> partitionLabels(String s) {}

  /**
   * 罗马数字转整数
   *
   * <p>扩展1，汉字转阿拉伯数字
   *
   * <p>扩展2，IP 与 integer 互转，参下
   *
   * @param s the s
   * @return int int
   */
  public int romanToInt(String s) {
    Map<Character, Integer> hash =
        new HashMap<Character, Integer>() {
          {
            //            put('一', 1);
            //            put('二', 2);
            //            put('三', 3);
            //            put('四', 4);
            //            put('五', 5);
            //            put('六', 6);
            //            put('七', 7);
            //            put('八', 8);
            //            put('九', 9);
            //            put('十', 10);
            //            put('百', 100);
            //            put('千', 1000);
            //            put('万', 10000);
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
          }
        };
    int res = 0;
    for (int i = 0; i < s.length(); i++) {
      int num = hash.get(s.charAt(i));
      if (i < s.length() - 1 && num < hash.get(s.charAt(i + 1))) res -= num;
      else res += num;
    }
    return res;
  }

  /**
   * 整数转罗马数字，greedy 尽可能先选出大的数字进行转换
   *
   * <p>扩展1，阿拉伯数字转汉字，数字先一一对应建映射，逢位加十百千万标识
   *
   * @param num the num
   * @return string string
   */
  public String intToRoman(int num) {
    final int[] NUMs = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    final String[] ROMANs = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    StringBuilder res = new StringBuilder();
    int cur = num;
    for (int i = 0; i < ROMANs.length; i++) {
      while (cur >= NUMs[i]) {
        res.append(ROMANs[i]);
        cur -= NUMs[i];
      }
    }
    return res.toString();
  }

  public int ipToInt(String s) {
    String[] ipList = s.split("\\.");
    int res = 0;
    for (String seg : ipList) {
      res = res << 8 | Integer.parseInt(seg, 10);
    }
    return res;
  }

  public String intToIP(int num) {
    String res = "";

    return res;
  }
}

/** 数学类 */
class MMath {
  final String CHARS = "0123456789ABCDEF";

  /**
   * 跳跃游戏，判断能否到达最后一个格，每格的数值表示可选的上界
   *
   * @param nums the nums
   * @return boolean boolean
   */
  public boolean canJump(int[] nums) {
    int furthest = 0; // 前 n-1 个元素能够跳到的最远距离
    for (int i = 0; i <= furthest; i++) {
      int curFurthest = i + nums[i]; // 第 i 个元素能够跳到的最远距离
      furthest = Math.max(furthest, curFurthest); // 更新最远距离
      // 如果最远距离已经大于或等于最后一个元素的下标，则说明能跳过去，结束
      if (furthest >= nums.length - 1) {
        return true;
      }
    }
    return false; // 最远距离 k 不再改变,且没有到末尾元素
  }

  /**
   * 跳跃游戏 II，返回到达最后一位到最少跳跃数
   *
   * <p>分别记录第 res+1 步可以到达的上下界，直到上界超过终点即结束迭代，此时的步数即为最少
   *
   * @param nums the nums
   * @return int int
   */
  public int jump(int[] nums) {
    int res = 0;
    int curLo = 0, curHi = 0;
    while (curHi < nums.length - 1) {
      int tmp = 0;
      for (int i = curLo; i <= curHi; i++) {
        tmp = Math.max(nums[i] + i, tmp);
      }
      curLo = curHi + 1;
      curHi = tmp;
      res += 1;
    }
    return res;
  }

  /**
   * x的平方根，二分只能精确至后二位
   *
   * <p>此处必须 /2 而非 >>1 比如，在区间只有 22 个数的时候，本题 if、else 的逻辑区间的划分方式是：[left..mid - 1] 与 [mid..right]
   *
   * <p>如果 mid 下取整，在区间只有 22 个数的时候有 mid 的值等于 left，一旦进入分支 [mid..right] 区间不会再缩小，出现死循环
   *
   * <p>扩展1，精确至 k 位，只能使用牛顿迭代法，参考
   * https://leetcode-cn.com/problems/sqrtx/solution/niu-dun-die-dai-fa-by-loafer/
   *
   * <p>扩展2，误差小于 1*10^(-k)，即精确至 k+1 位，同上
   *
   * @param x the x
   * @return int int
   */
  public int mySqrt(int x) {
    if (x == 0 || x == 1) {
      return x;
    }
    int lo = 0, hi = x, res = 0;
    while (lo <= hi) {
      int mid = lo + (hi - lo) / 2;
      if (mid > x / mid) {
        hi = mid - 1;
      } else {
        res = mid;
        lo = mid + 1;
      }
    }
    //    if (x == 0) {
    //      return 0;
    //    }
    //    double ori = x, res = x;
    //    while (true) {
    //      double xi = 0.5 * (res + ori / res);
    //      // 1e-7
    //      if (Math.abs(res - xi) < Integer.MIN_VALUE) break;
    //      res = xi;
    //    }
    //    return (int) res;

    return res;
  }

  /**
   * Pow(x,n)，快速幂
   *
   * @param x the x
   * @param n the n
   * @return double double
   */
  public double myPow(double x, int n) {
    return n >= 0 ? quickMulti(x, n) : 1.0 / quickMulti(x, -n);
  }

  private double quickMulti(double x, int n) {
    if (n == 0) return 1; // 1.剩余 0 次幂
    double y = quickMulti(x, n / 2); // 2.dichotomy
    return y * y * (((n & 1) == 0) ? 1 : x); // 3.剩余奇 or 偶次幂
  }

  /**
   * 圆圈中最后剩下的数字，约瑟夫环 Josephus Problem
   *
   * <p>记住公式即可 res = (res+m)%i
   *
   * @param n the n
   * @param m the m
   * @return the int
   */
  public int lastRemaining(int n, int m) {
    int res = 0;
    for (int i = 2; i <= n; i++) res = (res + m) % i;
    return res;
  }

  /**
   * rand7生成rand10 即[1,10]
   *
   * <p>等同进制转换的思路
   *
   * <p>数学推论，记住即可 (randX-1)*Y+randY() -> 等概率[1,X*Y]，只要 rand_N() 中 N 是 2 的倍数，就都可以用来实现 rand2()
   *
   * <p>扩展，给一个硬币，求生成 1/5 的概率同理即 rand2 -> rand5
   *
   * @return the int
   */
  public int rand10() {
    while (true) {
      int num = (rand7() - 1) * 7 + rand7(); // 等概率生成 [1,49] 范围的随机数
      if (num <= 40) return num % 10 + 1; // 拒绝采样，并返回 [1,10] 范围的随机数
    }
  }

  private int rand7() {
    return 0;
  }

  /**
   * 进制转换，除 radix 取余 & 倒排 & 高位补零，参考大数相加
   *
   * <p>https://www.nowcoder.com/practice/2cc32b88fff94d7e8fd458b8c7b25ec1?tpId=196&tqId=37170&rp=1&ru=/exam/oj&qru=/exam/oj&sourceUrl=%2Fexam%2Foj%3Ftab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D196%26page%3D1&difficulty=undefined&judgeStatus=undefined&tags=&title=
   *
   * @param num
   * @param radix
   * @return
   */
  public String baseConvert(int num, int radix) {
    if (num == 0) {
      return "0";
    }
    StringBuilder res = new StringBuilder();
    boolean f = false;
    if (num < 0) {
      f = true;
      num = -num;
    }
    while (num != 0) {
      res.append(CHARS.charAt(num % radix));
      num /= radix;
    }
    if (f) {
      res.append("-");
    }
    return res.reverse().toString();
  }

  /**
   * 第n位数字
   *
   * <p>一位数 9 个 -> 1 * 9，二位数 90 个 -> 2 * 90，其余同理
   *
   * @param n the n
   * @return int int
   */
  public int findNthDigit(int n) {
    int cur = 1, base = 9, k = n;
    while (k > cur * base) {
      k -= cur * base;
      cur += 1;
      base *= 10;
      if (cur > Integer.MAX_VALUE / base) break;
    }
    k -= 1;
    int num = (int) Math.pow(10, cur - 1) + k / cur;
    int idx = k % cur;
    return num / (int) Math.pow(10, cur - 1 - idx) % 10;
  }

  /**
   * 多数元素，摩尔投票，类比 Raft
   *
   * <p>尽管不通用，但对于本题方便理解和记忆
   *
   * <p>如果候选人是 maj , 则 maj 会支持自己，其他候选人会反对，当其为众数时其票数会过半，所以 maj 一定会成功当选
   *
   * @param nums the nums
   * @return int
   */
  public int majorityElement(int[] nums) {
    // 当前遍历的元素即 candidate 及其个数即 votes
    int num = nums[0], count = 1;
    for (int i = 1; i < nums.length; ++i) {
      if (count == 0) {
        num = nums[i];
        count = 1;
      } else if (nums[i] == num) {
        count += 1;
      } else {
        count -= 1;
      }
    }
    return num;
  }

  /**
   * 整数反转，32 位
   *
   * @param x the x
   * @return int int
   */
  public int reverse(int x) {
    // 暂存已反转的和剩余待反转的部分
    int res = 0, left = x;
    while (left != 0) {
      // 判断是否溢出 32 位整数
      if (res < Integer.MIN_VALUE / 10 || res > Integer.MAX_VALUE / 10) return 0;
      // 每次取末尾数字
      int tail = left % 10;
      left /= 10;
      res = res * 10 + tail;
    }
    return res;
  }

  /**
   * 回文数
   *
   * <p>0.特判负数 & 个位为 0
   *
   * <p>1.为取出 x 的左右部分，每次进行取余操作取出最低的数字，并加到取出数的末尾
   *
   * <p>2.判断左右部分是否数值相等 or 位数为奇数时右部分去最高位
   *
   * @param x the x
   * @return boolean boolean
   */
  public boolean isPalindrome(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) {
      return false;
    }
    int left = x, right = 0;
    while (left > right) {
      right = right * 10 + left % 10;
      left /= 10;
    }
    return left == right || left == right / 10;
  }

  /**
   * 颠倒二进制位，题设 32 位下，对于 Java need treat n as an unsigned value
   *
   * <p>归并，两位互换 -> 4 -> 8 -> 16
   *
   * @param n the n
   * @return int int
   */
  public int reverseBits(int n) {
    // 01010101010101010101010101010101
    n = ((n & 0xAAAAAAAA) >>> 1) | ((n & 0x55555555) << 1);
    // 00110011001100110011001100110011
    n = ((n & 0xCCCCCCCC) >>> 2) | ((n & 0x33333333) << 2);
    // 00001111000011110000111100001111
    n = ((n & 0xF0F0F0F0) >>> 4) | ((n & 0x0F0F0F0F) << 4);
    // 00000000111111110000000011111111
    n = ((n & 0xFF00FF00) >>> 8) | ((n & 0x00FF00FF) << 8);
    // 00000000000000001111111111111111
    n = ((n & 0xFFFF0000) >>> 16) | ((n & 0x0000FFFF) << 16);
    return n;
  }

  /**
   * 位1的个数，you need to treat n as an unsigned value
   *
   * <p>逐位与判断即可
   *
   * @param n the n
   * @return int int
   */
  public int hammingWeight(int n) {
    int res = 0;
    while (n != 0) {
      res += n & 1;
      n >>>= 1;
    }
    return res;
  }

  /**
   * 只出现一次的数字，只出现一次的字符参考 SString
   *
   * @param nums the nums
   * @return int int
   */
  public int singleNumber(int[] nums) {
    int res = 0;
    for (int num : nums) res ^= num;
    return res;
  }
}

/** 构建新数据结构 */
class DData {
  /**
   * LRU缓存机制
   *
   * <p>hash 保证 O(1) 寻址 & 链表保证 DML 有序
   *
   * <p>双向保证将一个节点移到双向链表的头部，可以分成「删除该节点」和「在双向链表的头部添加节点」两步操作，都可以在 O(1) 时间内完成
   *
   * <p>扩展1，处理输入输出
   *
   * <p>扩展2，线程安全，空结点 throw exception，分别对 hash & 双向链表改用 ConcurrentHashMap & 读写锁，前者可以使用另一把锁代替
   *
   * <p>扩展3，带超时
   */
  public class LRUCache {
    private final Map<Integer, DLinkedNode> cache = new HashMap<Integer, DLinkedNode>();
    private final int capacity;
    private final DLinkedNode head, tail; // dummy
    private int size;
    /**
     * Instantiates a new Lru cache.
     *
     * @param capacity the capacity
     */
    public LRUCache(int capacity) {
      this.capacity = capacity;
      head = new DLinkedNode();
      tail = new DLinkedNode();
      head.next = tail;
      tail.prev = head;
      //      ReadWriteLock lock = new ReentrantReadWriteLock();
      //      rl = lock.readLock();
      //      wl = lock.writeLock();
    }
    //    private final Lock rl, wl;

    /**
     * get & moveToHead
     *
     * @param key the key
     * @return the int
     */
    public int get(int key) {
      DLinkedNode node = cache.get(key);
      if (node == null) {
        return -1;
      }
      moveToHead(node);
      return node.value;
    }

    /**
     * 1.有则 set & moveToHead，否则 put & addToHead
     *
     * <p>2.溢出则 removeTail
     *
     * @param key the key
     * @param value the value
     */
    public void put(int key, int value) {
      DLinkedNode node = cache.get(key);
      if (node != null) {
        node.value = value;
        moveToHead(node);
        return;
      }
      DLinkedNode newNode = new DLinkedNode(key, value);
      cache.put(key, newNode);
      addToHead(newNode);
      size += 1;
      if (size > capacity) {
        DLinkedNode tail = removeTail();
        cache.remove(tail.key);
        size -= 1;
      }
    }

    private void addToHead(DLinkedNode node) {
      node.prev = head;
      node.next = head.next;
      head.next.prev = node;
      head.next = node;
    }

    private void moveToHead(DLinkedNode node) {
      removeNode(node);
      addToHead(node);
    }

    private DLinkedNode removeTail() {
      DLinkedNode res = tail.prev;
      removeNode(res);
      return res;
    }

    private void removeNode(DLinkedNode node) {
      node.prev.next = node.next;
      node.next.prev = node.prev;
    }

    private class DLinkedNode {
      public int key, value;
      public DLinkedNode prev, next;
      /** Instantiates a new D linked node. */
      public DLinkedNode() {}

      /**
       * Instantiates a new D linked node.
       *
       * @param _key the key
       * @param _value the value
       */
      public DLinkedNode(int _key, int _value) {
        key = _key;
        value = _value;
      }
    }
  }

  /**
   * 最小栈
   *
   * @author cenghui
   */
  public class MinStack {
    private final Deque<Integer> stack = new ArrayDeque<>();
    private int min;

    /**
     * Push.
     *
     * @param x the x
     */
    public void push(int x) {
      if (stack.isEmpty()) min = x;
      stack.push(x - min); // 存差
      if (x < min) min = x; // 更新
    }

    /** Pop. */
    public void pop() {
      if (stack.isEmpty()) return;
      int pop = stack.pop();
      // 弹出的是负值，要更新 min
      if (pop < 0) min -= pop;
    }

    /**
     * Top int.
     *
     * @return the int
     */
    public int top() {
      int top = stack.peek();
      // 负数的话，出栈的值保存在 min 中，出栈元素加上最小值即可
      return top < 0 ? min : top + min;
    }

    /**
     * Gets min.
     *
     * @return the min
     */
    public int getMin() {
      return min;
    }
  }

  /**
   * 用栈实现队列，双栈，in & out，均摊可以认为时间复制度为 O(1)
   *
   * <p>记忆，out & in & out & in
   */
  public class MyQueue {
    private final Deque<Integer> out = new ArrayDeque<>(), in = new ArrayDeque<>();
    /**
     * Push.
     *
     * @param x the x
     */
    public void push(int x) {
      in.addLast(x);
    }

    /**
     * Pop int.
     *
     * @return the int
     */
    public int pop() {
      peek(); // 仅为复用
      return out.removeLast();
    }

    /**
     * Peek int.
     *
     * @return the int
     */
    public int peek() {
      if (out.isEmpty()) while (!in.isEmpty()) out.addLast(in.removeLast());
      return out.getLast();
    }

    /**
     * Empty boolean.
     *
     * @return the boolean
     */
    public boolean empty() {
      return out.isEmpty() && in.isEmpty();
    }
  }

  /**
   * 设计循环队列
   *
   * <p>front 指向队列头部，即首个有效数据的位置，而 rear 指向队尾下一个，即从队尾入队元素的位置
   */
  public class MyCircularQueue {
    private final int capacity;
    private final int[] data;
    private int front, rear;

    /**
     * Instantiates a new My circular queue.
     *
     * @param k the k
     */
    public MyCircularQueue(int k) {
      // 循环数组中任何时刻一定至少有一个位置不存放有效元素
      // 当 rear 循环到数组的前面，要从后面追上 front，还差一格的时候，判定队列为满
      capacity = k + 1;
      data = new int[capacity];
    }

    /**
     * En queue boolean.
     *
     * @param value the value
     * @return the boolean
     */
    public boolean enQueue(int value) {
      if (isFull()) return false;
      data[rear] = value;
      rear = (rear + 1) % capacity;
      return true;
    }

    /**
     * De queue boolean.
     *
     * @return the boolean
     */
    public boolean deQueue() {
      if (isEmpty()) return false;
      front = (front + 1) % capacity;
      return true;
    }

    /**
     * Front int.
     *
     * @return the int
     */
    public int Front() {
      return isEmpty() ? -1 : data[front];
    }

    /**
     * Rear int.
     *
     * @return the int
     */
    public int Rear() {
      return isEmpty() ? -1 : data[(rear - 1 + capacity) % capacity];
    }

    /**
     * Is empty boolean.
     *
     * @return the boolean
     */
    public boolean isEmpty() {
      return front == rear;
    }

    /**
     * Is full boolean.
     *
     * @return the boolean
     */
    public boolean isFull() {
      return (rear + 1) % capacity == front;
    }
  }
}

/**
 * LFU
 *
 * <p>参考
 * https://leetcode-cn.com/problems/lfu-cache/solution/chao-xiang-xi-tu-jie-dong-tu-yan-shi-460-lfuhuan-c
 * 即可
 */
class LFUCache {
  private final Map<Integer, Node> cache; // 存储缓存的内容
  private final Map<Integer, DoublyLinkedList> freqMap; // 存储每个频次对应的双向链表
  private final int capacity;
  int size, min; // 存储当前最小频次

  public LFUCache(int capacity) {
    cache = new HashMap<>(capacity);
    freqMap = new HashMap<>();
    this.capacity = capacity;
  }

  public int get(int key) {
    Node node = cache.get(key);
    if (node == null) {
      return -1;
    }
    freqInc(node);
    return node.value;
  }

  public void put(int key, int value) {
    if (capacity == 0) {
      return;
    }
    Node node = cache.get(key);
    if (node != null) {
      node.value = value;
      freqInc(node);
    } else {
      if (size == capacity) {
        DoublyLinkedList minFreqLinkedList = freqMap.get(min);
        cache.remove(minFreqLinkedList.tail.pre.key);
        // 这里不需要维护min, 因为下面add了newNode后min肯定是1.
        minFreqLinkedList.removeNode(minFreqLinkedList.tail.pre);
        size -= 1;
      }
      Node newNode = new Node(key, value);
      cache.put(key, newNode);
      DoublyLinkedList linkedList = freqMap.get(1);
      if (linkedList == null) {
        linkedList = new DoublyLinkedList();
        freqMap.put(1, linkedList);
      }
      linkedList.addNode(newNode);
      size += 1;
      min = 1;
    }
  }

  void freqInc(Node node) {
    // 从原freq对应的链表里移除, 并更新min
    int freq = node.freq;
    DoublyLinkedList linkedList = freqMap.get(freq);
    linkedList.removeNode(node);
    if (freq == min && linkedList.head.post == linkedList.tail) {
      min = freq + 1;
    }
    // 加入新freq对应的链表
    node.freq += 1;
    linkedList = freqMap.get(freq + 1);
    if (linkedList == null) {
      linkedList = new DoublyLinkedList();
      freqMap.put(freq + 1, linkedList);
    }
    linkedList.addNode(node);
  }
}

class Node {
  final int key;
  int value, freq;
  Node pre, post;

  public Node(int key, int value) {
    this.key = key;
    this.value = value;
    this.freq = 1;
  }
}

class DoublyLinkedList {
  Node head, tail;

  public DoublyLinkedList() {
    head = new Node(Integer.MAX_VALUE, 0);
    tail = new Node(Integer.MAX_VALUE, 0);
    head.post = tail;
    tail.pre = head;
  }

  void removeNode(Node node) {
    node.pre.post = node.post;
    node.post.pre = node.pre;
  }

  void addNode(Node node) {
    node.post = head.post;
    head.post.pre = node;
    head.post = node;
    node.pre = head;
  }
}
