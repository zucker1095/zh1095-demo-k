package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集非五大基本类型的
 *
 * <p>进制转换，两种类型，确定进制，如 26，自定义进制，如人民币与罗马数字
 *
 * @author cenghui
 */
public class OOthers {
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
        new HashMap<>() {
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
      int cur = hash.get(s.charAt(i)), nxt = hash.get(s.charAt(i + 1));
      if (i < s.length() - 1 && cur < nxt) {
        res -= cur;
      } else {
        res += cur;
      }
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

  /**
   * Excel表列序号，26 转十进制，类似罗马数字转整数
   *
   * @param columnTitle the column title
   * @return int int
   */
  public int titleToNumber(String ct) {
    int res = 0;
    for (char ch : ct.toCharArray()) {
      res = res * 26 + (ch - 'A' + 1);
    }
    return res;
  }

  /**
   * Excel表列名称，十进制转 26
   *
   * <p>一般进制转换无须进行额外操作，是因为我们是在「每一位数值范围在 [0,x)」的前提下进行「逢 x 进一」。
   *
   * <p>但本题需要我们将从 1 开始，因此在执行「进制转换」操作前，我们需要先对 cn 减一，从而实现整体偏移
   *
   * @param cn
   * @return
   */
  public String convertToTitle(int cn) {
    StringBuilder res = new StringBuilder();
    while (cn > 0) {
      cn -= 1;
      res.append((char) (cn % 26 + 'A'));
      cn /= 26;
    }
    res.reverse();
    return res.toString();
  }

  /**
   * 跳跃游戏，判断能否到达最后一个格，每格的数值表示可选的上界
   *
   * @param nums the nums
   * @return boolean boolean
   */
  public boolean canJump(int[] nums) {
    // 前 n-1 个元素能够跳到的最远距离
    int furthest = 0;
    for (int i = 0; i <= furthest; i++) {
      // 第 i 个元素能够跳到的最远距离
      int curFurthest = i + nums[i];
      // 更新最远距离
      furthest = Math.max(furthest, curFurthest);
      // 如果最远距离已经大于或等于最后一个元素的下标，则说明能跳过去，结束
      if (furthest >= nums.length - 1) {
        return true;
      }
    }
    // 最远距离 k 不再改变，且没有到末尾元素
    return false;
  }

  /**
   * 跳跃游戏 II，返回到达最后一位到最少跳跃数
   *
   * <p>分别记录第 res+1 步可以到达的上下界 & 直到上界超过终点即结束迭代，此时的步数即为最少
   *
   * @param nums the nums
   * @return int int
   */
  public int jump(int[] nums) {
    int step = 0;
    int curLo = 0, curHi = 0;
    while (curHi < nums.length - 1) {
      int tmp = 0;
      for (int i = curLo; i <= curHi; i++) {
        tmp = Math.max(nums[i] + i, tmp);
      }
      curLo = curHi + 1;
      curHi = tmp;
      step += 1;
    }
    return step;
  }

  /**
   * 划分字母区间
   *
   * <p>TODO
   *
   * @param s the s
   * @return list
   */
  //  public List<Integer> partitionLabels(String s) {}

  /**
   * 栈排序
   *
   * <p>TODO
   *
   * @param stack the stack
   * @return deque
   */
  public Deque<Integer> stackSort(Deque<Integer> stack) {
    Deque<Integer> tmp = new ArrayDeque<>();
    while (!stack.isEmpty()) {}
    return tmp;
  }
}

/** 数学类 */
class MMath {
  private final String CHARS = "0123456789ABCDEF";

  /**
   * rand7生成rand10 即[1,10]，等同进制转换的思路
   *
   * <p>https://www.cnblogs.com/ymjyqsx/p/9561443.html
   *
   * <p>数学推论，记住即可 (randX-1)*Y+randY() -> 等概率[1,X*Y]，只要 rand_N() 中 N 是 2 的倍数，就都可以用来实现 rand2()
   *
   * <p>扩展1，比如 randX to randY，有如下情况，本质均是找平方与倍数
   *
   * <p>rand2 to rand3 取平方再拒绝采样，即本题
   *
   * <p>rand2 to rand5 先通过 rand2 to rand3，再 rand3 to rand5，取平方再拒绝采样，即第一种情况
   *
   * <p>rand5 to rand3 自旋，直接取，这种拒绝概率不大
   *
   * <p>rand5 to rand2 判断是否为 2 的倍数，类似本题 rand49 to rand10
   *
   * @return the int
   */
  public int rand10() {
    while (true) {
      // 等概率生成 [1,49] 范围的随机数
      int num = (rand7() - 1) * 7 + rand7();
      // 拒绝采样，并返回 [1,10] 范围的随机数
      if (num <= 40) {
        return num % 10 + 1;
      }
    }
  }

  private int rand7() {
    return 0;
  }

  /**
   * x的平方根，二分只能精确至后二位，建议采用牛顿迭代
   *
   * <p>此处必须 /2 而非 >>1 比如，在区间只有 22 个数的时候，本题 if、else 的逻辑区间的划分方式是：[left..mid-1] 与 [mid..right]
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
    if (x == 0) return 0;
    double pre = x;
    while (true) {
      // 2*cur=pre+x/pre
      double cur = 0.5 * (pre + x / pre);
      // 后 n 位此处定制
      if (Math.abs(pre - cur) < 1e-7) break;
      pre = cur;
    }
    return (int) pre;
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

  // 特判零次幂 & 递归二分 & 判断剩余幂
  private double quickMulti(double x, int n) {
    if (n == 0) return 1;
    double y = quickMulti(x, n / 2);
    return y * y * (((n & 1) == 0) ? 1 : x);
  }

  /**
   * 圆圈中最后剩下的数字，约瑟夫环 Josephus Problem
   *
   * <p>记住公式即可 res=(res+m)%i
   *
   * @param n the n
   * @param m the m
   * @return the int
   */
  public int lastRemaining(int n, int m) {
    int res = 0;
    for (int i = 2; i <= n; i++) {
      res = (res + m) % i;
    }
    return res;
  }

  /**
   * 进制转换，除 radix 取余 & 倒排 & 高位补零，参考大数相加
   *
   * <p>https://www.nowcoder.com/practice/2cc32b88fff94d7e8fd458b8c7b25ec1?tpId=196&tqId=37170&rp=1&ru=/exam/oj&qru=/exam/oj&sourceUrl=%2Fexam%2Foj%3Ftab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D196%26page%3D1&difficulty=undefined&judgeStatus=undefined&tags=&title=
   *
   * @param num the num
   * @param radix the radix
   * @return string string
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
   * 1~n整数中1出现的次数 / 数字1的个数
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/mian-shi-ti-43-1n-zheng-shu-zhong-1-chu-xian-de-2/
   *
   * @param n the n
   * @return int int
   */
  public int countDigitOne(int n) {
    int res = 0;
    int high = n / 10, cur = n % 10, low = 0, digit = 1;
    while (high != 0 || cur != 0) {
      if (cur == 0) {
        res += high * digit;
      } else if (cur == 1) {
        res += high * digit + low + 1;
      } else {
        res += (high + 1) * digit;
      }
      low += cur * digit;
      high /= 10;
      cur = high % 10;
      digit *= 10;
    }
    return res;
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
   * 多数元素，摩尔投票，类比 Raft
   *
   * <p>尽管不通用，但对于本题方便理解和记忆
   *
   * <p>如果候选人是 maj , 则 maj 会支持自己，其他候选人会反对，当其为众数时其票数会过半，所以 maj 一定会成功当选
   *
   * @param nums the nums
   * @return int int
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
   * 阶乘后点零，记住即可
   *
   * <p>参考
   * https://leetcode-cn.com/problems/factorial-trailing-zeroes/solution/xiang-xi-tong-su-de-si-lu-fen-xi-by-windliang-3/
   *
   * @param n
   * @return
   */
  public int trailingZeroes(int n) {
    int count = 0;
    while (n > 0) {
      n /= 5;
      count += n;
    }
    return count;
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
      /** The Key. */
      public int key,
          /** The Value. */
          value;

      /** The Prev. */
      public DLinkedNode prev,
          /** The Next. */
          next;
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
   * 设计循环队列
   *
   * <p>front 指向队列头部，即首个有效数据的位置，而 rear 指向队尾下一个，即从队尾入队元素的位置
   *
   * <p>扩展1，并发安全，单个 push 多个 pop
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

  /** 设计哈希映射 */
  public class MyHashMap {
    private static final int BASE = 769;
    private LinkedList[] data;

    MyHashMap() {
      data = new LinkedList[BASE];
      for (int i = 0; i < BASE; i++) {
        data[i] = new LinkedList<Pair>();
      }
    }

    private int hash(int key) {
      return key % BASE;
    }

    public void put(int key, int value) {
      int h = hash(key);
      Iterator<Pair> iterator = data[h].iterator();
      while (iterator.hasNext()) {
        Pair pair = iterator.next();
        if (pair.key == key) {
          pair.value = value;
          return;
        }
      }
      data[h].offerLast(new Pair(key, value));
    }

    public int get(int key) {
      int h = hash(key);
      Iterator<Pair> iterator = data[h].iterator();
      while (iterator.hasNext()) {
        Pair pair = iterator.next();
        if (pair.key == key) {
          return pair.value;
        }
      }
      return -1;
    }

    public void remove(int key) {
      int h = hash(key);
      Iterator<Pair> iterator = data[h].iterator();
      while (iterator.hasNext()) {
        Pair pair = iterator.next();
        if (pair.key == key) {
          data[h].remove(pair);
          return;
        }
      }
    }

    private class Pair {
      final int key;
      int value;

      Pair(int key, int value) {
        this.key = key;
        this.value = value;
      }
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
}

/** 二进制，位运算相关，掌握常见的运算符即可，现场推，毕竟命中率超低 */
class BBit {
  /**
   * 整数反转，32 位
   *
   * @param x the x
   * @return int int
   */
  public int reverse(int x) {
    // 暂存已反转的 & 待反转的部分
    int res = 0, left = x;
    while (left != 0) {
      // 判断溢出
      if (res < Integer.MIN_VALUE / 10 || res > Integer.MAX_VALUE / 10) {
        return 0;
      }
      // 每次取末尾数字
      res = res * 10 + left % 10;
      left /= 10;
    }
    return res;
  }

  /**
   * 颠倒二进制位，反转整数的二进制位，题设 32 位下，对于 Java need treat n as an unsigned value
   *
   * <p>分治，两位互换 -> 4 -> 8 -> 16
   *
   * <p>记住 ff00ff00，f0f0f0f0 镜像与 c & 3，a & 5 即可
   *
   * @param n the n
   * @return int int
   */
  public int reverseBits(int n) {
    // 低 16 位与高 16 位交换
    n = (n >> 16) | (n << 16);
    // 每16位中低8位和高8位交换，1111 是 f
    n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8);
    // 每8位中低4位和高4位交换
    n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4);
    // 每4位中低2位和高2位交换，1100 是 c,0011 是 3
    n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2);
    // 每2位中低1位和高1位交换，1010 是 a，0101 是 5
    n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1);
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
   * 只出现一次的数字
   *
   * <p>扩展1，有序数组，参考「有序数组中的单一元素」
   *
   * <p>扩展2，只出现一次的字符，参考「第一个只出现一次的字符」
   *
   * @param nums the nums
   * @return int int
   */
  public int singleNumber(int[] nums) {
    int res = 0;
    for (int num : nums) {
      res ^= num;
    }
    return res;
  }
}

// 参考图片即可
// https://leetcode-cn.com/problems/lfu-cache/solution/chao-xiang-xi-tu-jie-dong-tu-yan-shi-460-lfuhuan-c/
// class LFUCache {
//  // key->Node 这种结构的哈希表
//  private final Map<Integer, Node> keyMap = new HashMap<>();
//  // freq->LinkedList 这种结构的哈希表
//  private final Map<Integer, LinkedList> freqMap = new HashMap<>();
//  // 缓存的最大容量
//  private final int capacity;
//  // 记录缓存中最低频率
//  private int minFreq = 0;
//
//  public LFUCache(int capacity) {
//    //		if(capacity<=0) {
//    //			throw new IllegalArgumentException();
//    //		}
//    this.capacity = capacity;
//  }
//
//  /**
//   * 获取一个元素，如果key不存在则返回-1，否则返回对应的value，同时更新被访问元素的频率
//   *
//   * @param key 要查找的关键字
//   * @return 如果没找到则返回-1，否则返回对应的value
//   */
//  public int get(int key) {
//    if (!this.keyMap.containsKey(key)) {
//      return -1;
//    }
//    Node node = this.keyMap.get(key);
//    this.increment(node);
//    return node.getValue();
//  }
//
//  /**
//   * 插入指定的key和value，如果key存在则更新value，同时更新频率， 如果key不存并且缓存满了，则删除频率最低的元素，并插入新元素。否则，直接插入新元素
//   *
//   * @param key 要插入的关键字
//   * @param value 要插入的值
//   */
//  public void put(int key, int value) {
//    if (this.keyMap.containsKey(key)) {
//      Node node = this.keyMap.get(key);
//      node.updateValue(value);
//      this.increment(node);
//    } else {
//      if (this.capacity == 0) {
//        return;
//      }
//      if (this.keyMap.size() == this.capacity) {
//        this.remoteMinFreqNode();
//      }
//      Node node = new Node(key, value, 1);
//      this.increment(node, true);
//      this.keyMap.put(key, node);
//    }
//  }
//
//  /**
//   * 更新节点的访问频率
//   *
//   * @param node 要更新的节点
//   */
//  private void increment(Node node) {
//    increment(node, false);
//  }
//
//  /**
//   * 更新节点的访问频率
//   *
//   * @param node 要更新的节点
//   * @param isNewNode 是否是新节点，新插入的节点和非新插入节点更新逻辑不同
//   */
//  private void increment(Node node, boolean isNewNode) {
//    if (isNewNode) {
//      this.minFreq = 1;
//      this.insertToLinkedList(node);
//    } else {
//      this.deleteNode(node);
//      node.incrFreq();
//      this.insertToLinkedList(node);
//      if (!this.freqMap.containsKey(this.minFreq)) {
//        ++this.minFreq;
//      }
//    }
//  }
//
//  /**
//   * 根据节点的频率，插入到对应的LinkedList中，如果LinkedList不存在则创建
//   *
//   * @param node 将要插入到LinkedList的节点
//   */
//  private void insertToLinkedList(Node node) {
//    if (!this.freqMap.containsKey(node.getFreq())) {
//      this.freqMap.put(node.getFreq(), new LinkedList());
//    }
//    LinkedList linkedList = this.freqMap.get(node.getFreq());
//    linkedList.insertFirst(node);
//  }
//
//  /**
//   * 删除指定的节点，如果节点删除后，对应的双链表为空，则从__freqMap中删除这个链表
//   *
//   * @param node 将要删除的节点
//   */
//  private void deleteNode(Node node) {
//    LinkedList linkedList = this.freqMap.get(node.getFreq());
//    linkedList.deleteNode(node);
//    if (linkedList.isEmpty()) {
//      this.freqMap.remove(node.getFreq());
//    }
//  }
//
//  /** 删除频率最低的元素，从freqMap和keyMap中都要删除这个节点， 如果节点删除后对应的链表为空，则要从__freqMap中删除这个链表 */
//  private void remoteMinFreqNode() {
//    LinkedList linkedList = this.freqMap.get(this.minFreq);
//    Node node = linkedList.getLastNode();
//    linkedList.deleteNode(node);
//    this.keyMap.remove(node.getKey());
//    if (linkedList.isEmpty()) {
//      this.freqMap.remove(node.getFreq());
//    }
//  }
//
//  /** 双链表中的链表节点对象 */
//  protected class Node {
//    // 对应输入的key
//    private final int key;
//    // 指向前一个节点的指针
//    protected Node pre, next;
//    // 对应输入的value
//    private int value, freq;
//
//    public Node(int key, int value, int freq) {
//      this.key = key;
//      this.value = value;
//      this.freq = freq;
//    }
//  }
//
//  /** 自定义的双向链表类 */
//  protected class LinkedList {
//    // 双向链表的头结点
//    private final Node head, tail;
//
//    public LinkedList() {
//      this.head = Node.createEmptyNode();
//      this.tail = Node.createEmptyNode();
//      this.head.next = this.tail;
//      this.tail.pre = this.head;
//    }
//
//    /**
//     * 将指定的节点插入到链表的第一个位置
//     *
//     * @param node 将要插入的节点
//     */
//    public void insertFirst(Node node) {
//      if (node == null) {
//        throw new IllegalArgumentException();
//      }
//      node.next = this.head.next;
//      this.head.next.pre = node;
//      node.pre = this.head;
//      this.head.next = node;
//    }
//
//    /**
//     * 从链表中删除指定的节点
//     *
//     * @param node 将要删除的节点
//     */
//    public void deleteNode(Node node) {
//      if (node == null) {
//        throw new IllegalArgumentException();
//      }
//      node.pre.next = node.next;
//      node.next.pre = node.pre;
//      node.pre = null;
//      node.next = null;
//    }
//
//    /**
//     * 从链表中获取最后一个节点
//     *
//     * @return 双向链表中的最后一个节点，如果是空链表则返回None
//     */
//    public Node getLastNode() {
//      if (this.head.next == this.tail) {
//        return Node.createEmptyNode();
//      }
//      return this.tail.pre;
//    }
//
//    /**
//     * 判断链表是否为空，除了head和tail没有其他节点即为空链表
//     *
//     * @return 链表不空返回True，否则返回False
//     */
//    public boolean isEmpty() {
//      return this.head.next == this.tail;
//    }
//  }
// }
