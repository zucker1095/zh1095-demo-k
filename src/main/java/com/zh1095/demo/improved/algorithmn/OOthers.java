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
   * 范围求和II，求所有 op 交集的最大面积
   *
   * <p>每次均对于 [0,a] & [0,b] 操作，因此最大值必出现在 (0,0)，问题转换为，什么范围内的数与位置 (0,0) 上的值相等，即什么范围会被每一次操作所覆盖
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/range-addition-ii/solution/gong-shui-san-xie-jian-dan-mo-ni-ti-by-a-006h/
   *
   * @param m
   * @param n
   * @param ops
   * @return
   */
  public int maxCount(int m, int n, int[][] ops) {
    int minX = m, minY = n;
    for (int[] op : ops) {
      minX = Math.min(minX, op[0]);
      minY = Math.min(minY, op[1]);
    }
    return minX * minY;
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

/** 计算 */
class MMath {
  /**
   * rand7生成rand10 即[1,10]，等同进制转换的思路
   *
   * <p>参考
   * https://leetcode-cn.com/problems/implement-rand10-using-rand7/solution/cong-pao-ying-bi-kai-shi-xun-xu-jian-jin-ba-zhe-da/
   * https://www.cnblogs.com/ymjyqsx/p/9561443.html
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
   * <p>扩展2，一个两面不均匀的硬币，正面概率 p，反面概率 1-p，怎么保证通过抛硬币的结果来实现 0 1 的均匀分布
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
   * 1~n整数中1出现的次数 / 数字1的个数，如 1～12 这些整数中包含 1 的数字有 1、10、11、12 共 5 个
   *
   * <p>因此，将 1~n 的个位、十位、百位...1 出现次数相加，即为所求
   *
   * <p>将 x 位数的 n 分为 nx...ni...n1，则记 nx...ni+1 为 high，ni 为 cur，剩余为 low，10^i 为 digit
   *
   * <p>cur 分三种情况讨论，1 由高低一起决定，0 & else 则出现次数由高位决定
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/solution/mian-shi-ti-43-1n-zheng-shu-zhong-1-chu-xian-de-2/
   *
   * @param n the n
   * @return int int
   */
  public int countDigitOne(int n) {
    int count = 0;
    int high = n / 10, cur = n % 10, low = 0, digit = 1;
    while (high != 0 || cur != 0) {
      // 状态
      if (cur == 0) count += high * digit;
      else if (cur == 1) count += high * digit + low + 1;
      else count += (high + 1) * digit;
      // 递推
      low += cur * digit;
      cur = high % 10;
      high /= 10;
      digit *= 10;
    }
    return count;
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
    double pre = x, cur = 0;
    while (true) {
      // 2*cur = pre+x/pre
      cur = (pre + x / pre) * 0.5;
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
   * 平方数之和
   *
   * @param c
   * @return
   */
  public boolean judgeSquareSum(int c) {
    long a = 0, b = (long) Math.sqrt(c);
    while (a <= b) {
      long cur = a * a + b * b;
      if (cur < c) a += 1;
      else if (cur == c) return true;
      else if (cur > c) b -= 1;
    }
    return false;
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
   * 圆圈中最后剩下的数字，约瑟夫环 Josephus Problem
   *
   * <p>记住公式即可 res=(res+m)%i
   *
   * @param n the n
   * @param m the m
   * @return the int
   */
  public int lastRemaining(int n, int m) {
    int leftIdx = 0;
    for (int i = 2; i <= n; i++) {
      leftIdx = (leftIdx + m) % i;
    }
    return leftIdx;
  }

  /**
   * 阶乘后的零，求出一个数的阶乘末尾零的个数，如 5!=120 即 1
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
    private final Map<Integer, DLinkedNode> cache = new HashMap<>();
    // dummy
    private final DLinkedNode head = new DLinkedNode(), tail = new DLinkedNode();
    private final int capacity;
    /**
     * Instantiates a new Lru cache.
     *
     * @param capacity the capacity
     */
    public LRUCache(int capacity) {
      this.capacity = capacity;
      head.next = tail;
      tail.prev = head;
    }

    /**
     * get & moveToHead
     *
     * @param key the key
     * @return the int
     */
    public int get(int key) {
      DLinkedNode node = cache.get(key);
      if (node == null) return -1;
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
      if (cache.size() > capacity) {
        DLinkedNode tail = removeTail();
        cache.remove(tail.key);
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
    private int front, rear; // 虚拟头尾

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
      data[rear] = value; // CAS
      rear = (rear + 1) % capacity; // CAS
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
   * <p>全局保存最小值，入栈存差并更新，出栈与取顶均需判负
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
   * 设计哈希映射
   *
   * <p>扩展1，要求布隆过滤器
   */
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

/** 单个数字 */
class Digit {
  private final String CHARS = "0123456789ABCDEF";

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
   * <p>0.特判负数与十的倍数
   *
   * <p>1.为取出 x 的左右部分，每次进行取余操作取出最低的数字，并加到取出数的末尾
   *
   * <p>2.判断左右部分是否数值相等 or 位数为奇数时右部分去最高位
   *
   * @param x the x
   * @return boolean boolean
   */
  public boolean isPalindrome(int x) {
    if (x < 0 || (x % 10 == 0 && x != 0)) return false;
    // 左右部分
    int lo = x, hi = 0;
    while (lo > hi) {
      hi = hi * 10 + lo % 10;
      lo /= 10;
    }
    return lo == hi || lo == hi / 10;
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
   * 只出现一次的数字II，其余均出现三次
   *
   * <p>利用 int 固定为 32 bit，使用一个长度为 32 的数组 cnt[] 记录下所有数值的每一位共出现了多少次 1，再对 cnt[] 数组的每一位进行 mod 3
   * 操作，重新拼凑出只出现一次的数
   *
   * <p>如果一个数字出现三次,那么它的二进制表示的每一位(0或者1)也出现三次。如果把所有出现三次的数字的二进制表示的每一位都分别加起来,那么每一位的和都能被3整除
   *
   * <p>如果某一位的和能被3整除,那么那个只出现一次的数字二进制表示中对应的那一位是0;否则就是1
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/solution/javashi-xian-jian-zhi-si-lu-wei-yun-suan-zhu-wei-t/F
   *
   * <p>扩展1，有序数组，参考「有序数组中的单一元素」
   *
   * <p>扩展2，只出现一次的字符，参考「第一个只出现一次的字符」
   *
   * @param nums the nums
   * @return int int
   */
  public int singleNumber(int[] nums) {
    if (nums.length < 1) return -1;
    int[] bitSum = new int[32];
    for (int num : nums) {
      int bitMask = 1;
      // 首位为符号位
      for (int i = 31; i >= 0; i--) {
        if ((num & bitMask) != 0) bitSum[i] += 1;
        bitMask <<= 1;
      }
    }
    int target = 0;
    // 这种做法使得本算法同样适用于负数的情况
    for (int i = 0; i < 32; i++) {
      target += bitSum[i] % 3;
      target <<= 1;
    }
    return target;
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
    int count = 0;
    while (n != 0) {
      count += n & 1;
      n >>>= 1;
    }
    return count;
  }
}

// LFU 参考图片即可
// https://leetcode-cn.com/problems/lfu-cache/solution/chao-xiang-xi-tu-jie-dong-tu-yan-shi-460-lfuhuan-c/
