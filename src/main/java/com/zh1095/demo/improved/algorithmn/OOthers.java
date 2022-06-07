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
   * 加油站，解法类似「最大子序和」，本题环 & 求该子序和的起始位
   *
   * @param gas the gas
   * @param cost the cost
   * @return int int
   */
  public int canCompleteCircuit(int[] gas, int[] cost) {
    // 将问题转化为找最大子串的起始位置
    int startIdx = 0, sumGas = 0, leftGas = 0;
    for (int i = 0; i < gas.length; i++) {
      int curLeft = gas[i] - cost[i];
      leftGas += curLeft;
      if (sumGas > 0) {
        sumGas += curLeft;
      } else {
        sumGas = curLeft;
        startIdx = i;
      }
    }
    return leftGas >= 0 ? startIdx : -1;
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
   * 范围求和II，求所有 op 交集的最大面积
   *
   * <p>每次均对于 [0,a] & [0,b] 操作，因此最大值必出现在 (0,0)，问题转换为，什么范围内的数与位置 (0,0) 上的值相等，即什么范围会被每一次操作所覆盖
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/range-addition-ii/solution/gong-shui-san-xie-jian-dan-mo-ni-ti-by-a-006h/
   *
   * @param m the m
   * @param n the n
   * @param ops the ops
   * @return int int
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
   * 把数字翻译成字符串，返回方案数
   *
   * <p>以 xyzcba 为例，先取最后两位 即 ba，如果ba>=26，必然不能分解成 f(xyzcb)+f(xyzc)，此时只能分解成 f(xyzcb)
   *
   * <p>但还有一种情况 ba<=9 即也就是该数十位上为 0，也不能分解
   *
   * <p>参考
   * https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/solution/di-gui-qiu-jie-shuang-bai-by-xiang-shang-de-gua-ni/
   *
   * @param num
   * @return
   */
  public int translateNum(int num) {
    if (num <= 9) return 1;
    int ba = num % 100; // xyzcba
    return ba <= 9 || ba >= 26
        ? translateNum(num / 10)
        : translateNum(num / 10) + translateNum(num / 100);
  }

  /**
   * 排名百分比，根据成绩获取，要求 O(n) for time
   *
   * <p>counts 类似 bitmap，对 score 排序。
   *
   * <p>参考 https://leetcode.cn/circle/discuss/4ZX013/
   *
   * @param scores
   * @return
   */
  public double[] countScore(int[] scores) {
    final int MaxScore = 100; // 题设分数的上界
    // 以分数维度，前后两个循环共用一个数组，分别对应 equal count & lte count
    int[] counts = new int[MaxScore + 1];
    for (int s : scores) {
      counts[s] += 1;
    }
    for (int s = 1; s < MaxScore; s++) {
      counts[s] += counts[s - 1];
    }
    // 以学生维度
    int count = scores.length;
    double[] pcts = new double[count];
    for (int i = 0; i < count; i++) {
      pcts[i] = 100.0 * counts[scores[i]] / count;
    }
    return pcts;
  }

  /**
   * 多边形周长等分
   *
   * <p>TODO 参考 https://codeantenna.com/a/nyBNVsVmD5
   *
   * @param points the points
   * @param k the k
   * @return point [ ]
   */
  public Point[] splitPerimeter(Point[] points, int k) {
    int len = points.length;
    double perimeter = 0;
    double[] edges = new double[len];
    // 计算周长
    for (int i = 0; i < len; i++) {
      edges[i] = getDist(points[i], points[(i + 1) % len]);
      perimeter += edges[i];
    }
    Point[] splitPoints = new Point[k];
    double avgLen = perimeter / k, curLen = 0; // 平均距离 & 某条边截取剩余的距离
    for (int v = 0; v < k; v++) { // 从第一条边开始找齐 k 个点
      for (int e = 0; e < len; e++) {
        curLen += edges[e];
        if (curLen < avgLen) continue;
        double needLen = curLen - avgLen; // 当前边需要截取的距离
        splitPoints[v] = getSplitPoint(points[e], points[(e + 1) % len], edges[e], needLen);
        curLen = needLen;
        break;
      }
    }
    return splitPoints;
  }

  // 返回两点的欧氏距离
  private double getDist(Point p1, Point p2) {
    return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
  }

  // 获取等分点的坐标
  private Point getSplitPoint(Point p1, Point p2, double dist, double len) {
    Point p = new Point();
    // 利用长度比例计算点的坐标
    p.x = len / dist * (p1.x - p2.x) + p2.x;
    p.y = len / dist * (p1.y - p2.y) + p2.y;
    return p;
  }

  /**
   * 栈排序
   *
   * <p>TODO
   *
   * @param stack the stack
   * @return deque deque
   */
  public Deque<Integer> stackSort(Deque<Integer> stack) {
    Deque<Integer> tmp = new ArrayDeque<>();
    while (!stack.isEmpty()) {}
    return tmp;
  }

  /**
   * 划分字母区间，类似「跳跃游戏」
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/partition-labels/solution/python-jiu-zhe-quan-guo-zui-cai-you-hua-dai-ma-by-/
   */
  public List<Integer> partitionLabels(String s) {
    int[] lastIdxes = new int[26];
    char[] chs = s.toCharArray();
    for (int i = 0; i < chs.length; i++) {
      lastIdxes[chs[i] - 'a'] = i; // 题设均 lowercase
    }
    List<Integer> lens = new ArrayList<>();
    int lo = 0, hi = 0; // 当前片段的首尾
    for (int i = 0; i < chs.length; i++) {
      hi = Math.max(lastIdxes[chs[i] - 'a'], hi);
      if (i == hi) {
        lens.add(hi - lo + 1);
        lo = hi + 1;
      }
    }

    return lens;
  }

  // 「多边形周长等分」
  private class Point {
    /** The X. */
    double x,
        /** The Y. */
        y;

    /**
     * Instantiates a new Point.
     *
     * @param x the x
     * @param y the y
     */
    public Point(double x, double y) {
      this.x = x;
      this.y = y;
    }

    public Point() {}
  }
}

/** 构建新数据结构 */
class DData {
  /**
   * LRU缓存机制
   *
   * <p>hash 保证 O(1) 寻址 & 链表保证 DML 有序，双向保证将一个节点移到双向链表的头部，可以分成「删除该节点」&「在双向链表的头部添加节点」两步操作都 O(1)
   *
   * <p>To be implemented: addToHead, moveToHead, removeTail, removeTail.
   *
   * <p>扩展1，带超时，可以懒删除或 Daemon 随机 scan
   *
   * <p>扩展2，线程安全，空结点 throw exception，分别对 hash & 双向链表改用 ConcurrentHashMap & 读写锁，前者可以使用另一把锁代替
   */
  public class LRUCache {
    private final Map<Integer, DLinkedNode> key2Node = new HashMap<>();
    private final DLinkedNode head = new DLinkedNode(), tail = new DLinkedNode(); // dummy
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
      DLinkedNode node = key2Node.get(key);
      if (node == null) return -1;
      moveToHead(node);
      return node.value;
    }

    /**
     * 有则 set & moveToHead，否则 put & addToHead
     *
     * <p>溢出则 removeTail
     *
     * @param key the key
     * @param value the value
     */
    public void put(int key, int value) {
      DLinkedNode node = key2Node.get(key);
      if (node != null) {
        node.value = value;
        moveToHead(node);
        return;
      }
      DLinkedNode newNode = new DLinkedNode(key, value);
      key2Node.put(key, newNode);
      addToHead(newNode);
      if (key2Node.size() > capacity) {
        DLinkedNode tail = removeTail();
        key2Node.remove(tail.key);
      }
    }

    private void addToHead(DLinkedNode node) {
      node.prev = head;
      node.next = head.next;
      head.next.prev = node;
      head.next = node;
    }

    // removeTail & addToHead
    private void moveToHead(DLinkedNode node) {
      removeNode(node);
      addToHead(node);
    }

    // revmoeNode
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
   * 设计循环队列，三种实现方式，冗余一个元素 / 边界标记 / 计数器，此处选用前者
   *
   * <p>front 指向队列头部，即首个有效数据的位置，而 rear 指向队尾下一个，即从队尾入队元素的位置
   *
   * <p>扩展1，并发安全，单个 push 多个 pop
   */
  public class MyCircularQueue {
    private final int CAPACITY;
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
      CAPACITY = k + 1;
      data = new int[CAPACITY];
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
      rear = (rear + 1) % CAPACITY; // CAS
      return true;
    }

    /**
     * De queue boolean.
     *
     * @return the boolean
     */
    public boolean deQueue() {
      if (isEmpty()) return false;
      front = (front + 1) % CAPACITY;
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
      return isEmpty() ? -1 : data[(rear - 1 + CAPACITY) % CAPACITY];
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
      return (rear + 1) % CAPACITY == front;
    }
  }

  /**
   * 最小栈，全局保存最小值，入栈存差并更新，出栈与取顶均需判负
   *
   * <p>参考 https://yeqown.xyz/2018/03/01/Stack%E5%AE%9E%E7%8E%B0O1%E7%9A%84Min%E5%92%8CMax/
   *
   * <p>https://www.cnblogs.com/Acx7/p/14617661.html
   *
   * <p>TODO 扩展1，O(1) 同时返回最大与最小
   *
   * @author cenghui
   */
  public class MinStack {
    private final Deque<Integer> stack = new ArrayDeque<>();
    private int min, max;

    /**
     * 存差 & 更新
     *
     * @param x the x
     */
    public void push(int x) {
      if (stack.isEmpty()) min = x;
      stack.push(x - min);
      if (x < min) min = x;
    }

    /**
     * top < 0 ? min-=top
     *
     * <p>top > 0 ? max-=top
     */
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
      // 负数则出栈的值保存在 min 中，出栈元素加上最小值即可
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

    /**
     * Gets max.
     *
     * @return the max
     */
    public int getMax() {
      return max;
    }
  }

  /**
   * 设计哈希映射
   *
   * <p>扩展1，要求布隆过滤器
   */
  public class MyHashMap {
    private static final int CAPACITY = 769;
    private final List[] buckets;

    /** Instantiates a new My hash map. */
    public MyHashMap() {
      buckets = new LinkedList[CAPACITY];
      for (int i = 0; i < CAPACITY; i++) {
        buckets[i] = new LinkedList<BucketNode>();
      }
    }

    private int hash(int key) {
      return key % CAPACITY;
    }

    /**
     * Get int.
     *
     * @param key the key
     * @return the int
     */
    public int get(int key) {
      int h = hash(key);
      Iterator<BucketNode> iterator = buckets[h].iterator();
      while (iterator.hasNext()) {
        BucketNode pair = iterator.next();
        if (pair.key == key) {
          return pair.value;
        }
      }
      return -1;
    }

    /**
     * Put.
     *
     * @param key the key
     * @param value the value
     */
    public void put(int key, int value) {
      int h = hash(key);
      Iterator<BucketNode> iterator = buckets[h].iterator();
      while (iterator.hasNext()) {
        BucketNode pair = iterator.next();
        if (pair.key == key) {
          pair.value = value;
          return;
        }
      }
      buckets[h].add(new BucketNode(key, value));
    }

    /**
     * Remove.
     *
     * @param key the key
     */
    public void remove(int key) {
      int h = hash(key);
      Iterator<BucketNode> iterator = buckets[h].iterator();
      while (iterator.hasNext()) {
        BucketNode pair = iterator.next();
        if (pair.key == key) {
          buckets[h].remove(pair);
          return;
        }
      }
    }

    private class BucketNode {
      /** The Key. */
      final int key;

      /** The Value. */
      int value;

      /**
       * Instantiates a new Pair.
       *
       * @param key the key
       * @param value the value
       */
      public BucketNode(int key, int value) {
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

  /**
   * 实现Trie（前缀树）
   *
   * <p>参考
   * https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/trie-tree-de-shi-xian-gua-he-chu-xue-zhe-by-huwt/
   */
  public class Trie {
    private final TireNode root = new TireNode(); // 哑结点

    /**
     * Insert.
     *
     * @param word the word
     */
    public void insert(String word) {
      TireNode cur = root;
      for (char ch : word.toCharArray()) {
        if (cur.next[ch - 'a'] == null) cur.next[ch - 'a'] = new TireNode();
        cur = cur.next[ch - 'a'];
      }
      cur.isEnd = true;
    }

    /**
     * Search boolean.
     *
     * @param word the word
     * @return the boolean
     */
    public boolean search(String word) {
      TireNode target = lookup(word);
      return target == null ? false : target.isEnd;
    }

    /**
     * Starts with boolean.
     *
     * @param prefix the prefix
     * @return the boolean
     */
    public boolean startsWith(String prefix) {
      return lookup(prefix) != null;
    }

    private TireNode lookup(String word) {
      TireNode cur = root;
      for (char ch : word.toCharArray()) {
        cur = cur.next[ch - 'a'];
        if (cur == null) return null;
      }
      return cur;
    }

    private class TireNode {
      private final TireNode[] next = new TireNode[26];
      private boolean isEnd = false;
    }
  }

  /**
   * O(1) 时间插入、删除和获取随机元素
   *
   * <p>不能「使用拒绝采样」&「在数组非结尾位置添增删元素」，因此需要申请一个足够大的数组
   *
   * <p>以 val 为键，数组下标 loc 为值
   *
   * <p>参考 https://leetcode.cn/problems/insert-delete-getrandom-o1/solution/by-ac_oier-tpex/
   */
  public class RandomizedSet {
    private final Random random = new Random();
    private final int[] nums = new int[200010];
    private final Map<Integer, Integer> map = new HashMap<>();
    private int idx = -1;

    public boolean insert(int val) {
      if (map.containsKey(val)) return false;
      idx += 1;
      nums[idx] = val;
      map.put(val, idx);
      return true;
    }

    public boolean remove(int val) {
      if (!map.containsKey(val)) return false;
      int idxDeleted = map.remove(val);
      if (idxDeleted != idx) map.put(nums[idx], idxDeleted);
      nums[idxDeleted] = nums[idx];
      idx -= 1;
      return true;
    }

    public int getRandom() {
      return nums[random.nextInt(idx + 1)];
    }
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
      if (num <= 40) return num % 10 + 1;
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
   * x的平方根，建议采用牛顿迭代，二分只能精确至后二位
   *
   * <p>记忆即可，new=(old+num/old)*0.5
   *
   * <p>此处必须 /2 而非 >>1 比如，在区间只有 22 个数的时候，本题 if、else 的逻辑区间的划分方式是：[left..mid-1] 与 [mid..right]
   *
   * <p>如果 mid 下取整，在区间只有 22 个数的时候有 mid 的值等于 left，一旦进入分支 [mid..right] 区间不会再缩小，出现死循环
   *
   * <p>扩展1，精确至 k 位或误差小于 1*10^(-k)，只能使用牛顿迭代法，参考
   * https://leetcode-cn.com/problems/sqrtx/solution/niu-dun-die-dai-fa-by-loafer/
   *
   * @param x the x
   * @return int int
   */
  public int mySqrt(int x) {
    if (x == 0) return 0;
    double pre = x, cur;
    while (true) {
      cur = (pre + x / pre) * 0.5;
      // 后 n 位 1e-n
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
   * 分数到小数，高精度除法
   *
   * <p>每一位小数通过 *10 再除余计算，而循环小数可以通过判断被除数如 1/7 有没有出现过，及其结果的左边界来判断。
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/fraction-to-recurring-decimal/solution/pythonjavajavascript-gao-jing-du-chu-fa-44td5/
   *
   * @param numerator 分子
   * @param denominator 分母
   * @return
   */
  public String fractionToDecimal(int numerator, int denominator) {
    // 1.判断正负
    int flag = (numerator > 0 && denominator > 0) || (numerator < 0 && denominator < 0) ? 1 : -1;
    long a = Math.abs((long) numerator), b = Math.abs((long) denominator);

    // 2.取出余数
    long quotient = a / b, remainder = a % b;
    if (remainder == 0) return String.valueOf(quotient * flag); // 整除
    StringBuilder num = new StringBuilder(String.valueOf(quotient)); // 先插入整数部分
    if (flag == -1) num.insert(0, '-'); // 正负
    num.append('.');
    int len = num.length(); // 小数之外的长度

    final int PRECISSION = 10000; // 精度可控
    // 3.保存某个被除数与原除数的结果
    Map<Long, Integer> quotients = new HashMap<>();
    for (int i = 0; i < PRECISSION; i++) {
      a = remainder * 10;
      quotient = a / b;
      remainder = a % b;

      // 4.该被除数已出现过，在该区间首末插入括号即可
      if (quotients.containsKey(a)) {
        num.insert(quotients.get(a).intValue(), '(');
        num.append(')');
        break;
      }

      num.append(quotient);
      quotients.put(a, i + len); // 该除数所得结果的左边界
      if (remainder == 0) break; // 入迭代前非整除
    }
    return num.toString();
  }

  /**
   * 第N位数字 / 第n个数字
   *
   * <p>k 位数共有 9*10^(k-1) 个数字，迭代试减 n 以确定 k 所在的整个数字
   *
   * <p>基于规律 [100, 999] 有 3*90*100 个数字 即 3*9*10^2
   *
   * <p>参考
   * https://leetcode-cn.com/problems/nth-digit/solution/gong-shui-san-xie-jian-dan-mo-ni-ti-by-a-w5wl/
   *
   * @param n the n
   * @return int int
   */
  public int findNthDigit(int n) {
    // 分别表示当前的位数 & 还剩下要找的位数。
    int len = 1, left = n;
    // 找到 n 所在的位数区间，len 位数共有 9*len*10^(len-1) 个完整的数字。
    while (9 * len * Math.pow(10, len - 1) < left) {
      left -= 9 * len * Math.pow(10, len - 1);
      len += 1;
    }
    // 此时 left 所在完整的数字的值上下界是 [10^(len-1),10^len-1]
    // 可推出目标所在的数字 (target−min+1)*len >= left
    double minNum = Math.pow(10, len - 1), targetNum = (minNum + left / len - 1);
    // 相较于 min 需要取的完整的数字
    int count = left - len * (left / len);
    // cur==0 表示答案为目标数字的最后一位，否则是其从左到右的第 left 位。
    return (int) (count == 0 ? targetNum % 10 : (targetNum + 1) / Math.pow(10, len - count) % 10);
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
    // 高低位
    int lo = 0, hi = x;
    while (lo < hi) {
      lo = lo * 10 + hi % 10;
      hi /= 10;
    }
    return hi == lo || hi == lo / 10;
  }

  /**
   * 平方数之和
   *
   * @param c the c
   * @return boolean boolean
   */
  public boolean judgeSquareSum(int c) {
    long n1 = 0, n2 = (long) Math.sqrt(c);
    while (n1 <= n2) {
      long cur = n1 * n1 + n2 * n2;
      if (cur < c) n1 += 1;
      else if (cur == c) return true;
      else if (c < cur) n2 -= 1;
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
   * @param n the n
   * @return int int
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

/**
 * 收集图相关，有如下题型
 *
 * <p>判断两点之间的连通性，拓扑排序
 *
 * <p>求两点之间的最短路径
 *
 * <p>求两点之间的路径总数
 *
 * <p>求两点之间的权值最小的路径
 */
class GGraph {
  // 「N叉树的直径」结果
  private int diameter = 0;

  // 分为存图与算法两步
  // 存图分为两种
  private void learn() {
    // 点数与边数
    int points, edges;

    // 1.邻接矩阵，适合边数较多的稠密图
    // int[][] matrix = new int[points][points];
    // 增加一条 a->b 权重为 c 的边
    // matrix[a][b] = c;

    // 2.邻接表，适合稀疏图，类似数组头插存储单链表
    //    int[] he = new int[points], // he 存储是某个节点所对应的边的集合（链表）的头结点
    //        e = new int[edges], // e 由于访问某一条边指向的节点
    //        ne = new int[edges], // ne 由于是以链表的形式进行存边，该数组就是用于找到下一条边
    //        w = new int[edges]; // w 记录某边的权重
  }

  /**
   * N叉树的直径
   *
   * <p>TODO 参考 https://www.nowcoder.com/questionTerminal/a77b4f3d84bf4a7891519ffee9376df3
   *
   * @param n the n
   * @param Edges the tree edge
   * @param EdgeValues the edge value
   * @return int int
   */
  public int diameterOfTree(int n, Interval[] Edges, int[] EdgeValues) {
    List<Edge>[] graph = new List[n];
    for (int i = 0; i < n; i++) {
      graph[i] = new ArrayList<>();
    }
    // 建图
    for (int i = 0; i < Edges.length; i++) {
      Interval interval = Edges[i];
      int value = EdgeValues[i];
      // 由于是无向图，所有每条边都是双向的
      graph[interval.start].add(new Edge(interval.end, value));
      graph[interval.end].add(new Edge(interval.start, value));
    }
    // 随机从一个节点开始 dfs，这里选择 0
    dfs13(graph, -1, 0);
    return diameter;
  }

  // 返回值为从 node 节点开始的最长深度
  private int dfs13(List<Edge>[] graph, int from, int to) {
    // 从节点开始的最大与第二深度
    int maxDepth = 0, secondMaxDepth = 0;
    for (Edge edge : graph[to]) {
      int neighbor = edge.end;
      if (neighbor == from) continue; // 防止返回访问父节点
      int depth = edge.weight + dfs13(graph, to, neighbor);
      if (depth > maxDepth) {
        secondMaxDepth = maxDepth;
        maxDepth = depth;
      } else if (depth > secondMaxDepth) {
        secondMaxDepth = depth;
      }
    }
    // maxDepth+secondMaxDepth 为以此节点为中心的直径
    diameter = Math.max(diameter, maxDepth + secondMaxDepth);
    return maxDepth;
  }

  /**
   * 课程表，check if DAG，拓扑排序
   *
   * <p>原理是对 DAG 的顶点进行排序，使得对每一条有向边 (u, v)，均有 u（在排序记录中）比 v 先出现。亦可理解为对某点 v 而言，只有当 v 的所有源点均出现了，v 才能出现
   *
   * <p>因此，需要入度数组 & 图 & 遍历队列三种数据结构，步骤如下
   *
   * <p>1.建图并记录入度
   *
   * <p>2.收集入度为 0 的点
   *
   * <p>3.拓扑排序，遍历队列，出队处理，并收集其入度为 0 的邻接点
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return boolean boolean
   */
  public boolean canFinish(int numCourses, int[][] prerequisites) {
    // 每个点的入度 & 邻接表存储图结构 & BFS 遍历
    int[] indegrees = new int[numCourses];
    List<List<Integer>> graph = new ArrayList<>(numCourses);
    for (int i = 0; i < numCourses; i++) {
      graph.add(new ArrayList<>());
    }
    Queue<Integer> queue = new LinkedList<>();
    // [0, 1] is 0->1
    for (int[] cp : prerequisites) {
      int toID = cp[0], fromID = cp[1];
      indegrees[toID] += 1;
      graph.get(fromID).add(toID);
    }
    // courseID range 0 from numCourses-1 incrementally
    for (int i = 0; i < numCourses; i++) {
      if (indegrees[i] == 0) queue.add(i);
    }
    // BFS TopSort
    while (!queue.isEmpty()) {
      int pre = queue.poll();
      numCourses -= 1; // handle it
      for (int cur : graph.get(pre)) { // traverse its adjacency
        indegrees[cur] -= 1;
        if (indegrees[cur] == 0) queue.add(cur);
      }
    }
    return numCourses == 0;
  }

  /**
   * 课程表II，上方新增记录即可，检测循环依赖同理
   *
   * <p>若存在循环依赖则返回空，否则返回可行的编译顺序
   *
   * <p>参考 https://mp.weixin.qq.com/s/pCRscwKqQdYYN7M1Sia7xA
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return int [ ]
   */
  public int[] findOrder(int numCourses, int[][] prerequisites) {
    int[] paths = new int[numCourses], indegrees = new int[numCourses];
    List<List<Integer>> graph = new ArrayList<>(numCourses);
    for (int i = 0; i < numCourses; i++) {
      graph.add(new ArrayList<>());
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int[] cp : prerequisites) {
      int toID = cp[0], fromID = cp[1];
      indegrees[toID] += 1;
      graph.get(fromID).add(toID);
    }
    for (int i = 0; i < numCourses; i++) {
      if (indegrees[i] == 0) queue.add(i);
    }
    // 当前结果集的元素个数，正好可作为下标
    int count = 0;
    while (!queue.isEmpty()) {
      int pre = queue.poll();
      paths[count] = pre;
      count += 1;
      for (int n : graph.get(pre)) {
        indegrees[n] -= 1;
        if (indegrees[n] == 0) queue.add(n);
      }
    }
    // 如果结果集中的数量不等于结点的数量，就不能完成课程任务，这一点是拓扑排序的结论
    return count == numCourses ? paths : new int[0];
  }

  /**
   * 网络延迟时间，到达所有结点的最短路径，floyd
   *
   * <p>参考
   * https://leetcode-cn.com/problems/network-delay-time/solution/gong-shui-san-xie-yi-ti-wu-jie-wu-chong-oghpz/
   *
   * @param times the times
   * @param n the n
   * @param k the k
   * @return int int
   */
  public int networkDelayTime(int[][] times, int n, int k) {
    int[][] matrix = new int[n][n];
    // 初始化邻接矩阵
    for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) {
        matrix[i][j] = matrix[j][i] = i == j ? 0 : Integer.MAX_VALUE;
      }
    }
    // 存图
    for (int[] t : times) {
      int u = t[0], v = t[1], c = t[2];
      matrix[u][v] = c;
    }
    // 最短路
    // floyd 基本流程为三层循环
    // 枚举中转点 - 枚举起点 - 枚举终点 - 松弛操作
    for (int p = 1; p <= n; p++) {
      for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
          matrix[i][j] = Math.min(matrix[i][j], matrix[i][p] + matrix[p][j]);
        }
      }
    }
    // 遍历答案
    int res = 0;
    for (int i = 1; i <= n; i++) {
      res = Math.max(res, matrix[k][i]);
    }
    return res >= Integer.MAX_VALUE / 2 ? -1 : res;
  }

  /**
   * 访问所有节点的最短路径
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/shortest-path-visiting-all-nodes/solution/gong-shui-san-xie-yi-ti-shuang-jie-bfs-z-6p2k/
   *
   * @param graph the graph
   * @return int
   */
  public int shortestPathLength(int[][] graph) {
    int n = graph.length, mask = 1 << n, INF = Integer.MAX_VALUE;
    // Floyd 求两点的最短路径
    int[][] dist = new int[n][n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        dist[i][j] = INF;
      }
    }
    for (int i = 0; i < n; i++) {
      for (int j : graph[i]) {
        dist[i][j] = 1;
      }
    }
    for (int k = 0; k < n; k++) {
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
        }
      }
    }

    // DP 过程，如果从 i 能够到 j 的话，使用 i 到 j 的最短距离（步长）来转移
    int[][] f = new int[mask][n];
    // 起始时，让所有状态的最短距离（步长）为正无穷
    for (int i = 0; i < mask; i++) {
      Arrays.fill(f[i], INF);
    }
    // 由于可以将任意点作为起点出发，可以将这些起点的最短距离（步长）设置为 0
    for (int i = 0; i < n; i++) f[1 << i][i] = 0;

    // 枚举所有的 state
    for (int state = 0; state < mask; state++) {
      // 枚举 state 中已经被访问过的点
      for (int i = 0; i < n; i++) {
        if (((state >> i) & 1) == 0) continue;
        // 枚举 state 中尚未被访问过的点
        for (int j = 0; j < n; j++) {
          if (((state >> j) & 1) == 1) continue;
          f[state | (1 << j)][j] = Math.min(f[state | (1 << j)][j], f[state][i] + dist[i][j]);
        }
      }
    }
    int ans = INF;
    for (int i = 0; i < n; i++) {
      ans = Math.min(ans, f[mask - 1][i]);
    }
    return ans;
  }

  // 「克隆图」
  private final Map<Node, Node> visited = new HashMap<>();

  /**
   * 克隆图，deep clone undirected connected graph，DFS
   *
   * @param node
   * @return
   */
  public Node cloneGraph(Node node) {
    if (node == null) return node;
    if (visited.containsKey(node)) return visited.get(node);
    // 为深拷贝我们不会克隆它的邻居的列表
    Node cloneNode = new Node(node.val, new ArrayList());
    visited.put(node, cloneNode);
    // 遍历该节点的邻居并更新克隆节点的邻居列表
    for (Node nbh : node.neighbors) {
      cloneNode.neighbors.add(cloneGraph(nbh));
    }
    return cloneNode;
  }

  private class Edge {
    /** The End. */
    int end,
        /** The Weight. */
        weight;

    /**
     * Instantiates a new Edge.
     *
     * @param end the end
     * @param w the w
     */
    Edge(int end, int w) {
      this.end = end;
      this.weight = w;
    }
  }

  private class Interval {
    /** The Start. */
    int start,
        /** The End. */
        end;
  }

  private class Node {
    public int val;
    public final List<Node> neighbors;

    public Node() {
      val = 0;
      neighbors = new ArrayList<Node>();
    }

    public Node(int _val) {
      val = _val;
      neighbors = new ArrayList<Node>();
    }

    public Node(int _val, ArrayList<Node> _neighbors) {
      val = _val;
      neighbors = _neighbors;
    }
  }
}

/** 二进制，位运算相关，掌握常见的运算符即可，现场推，毕竟命中率超低 */
class BBit {
  /**
   * 整数反转，反转十进制位。
   *
   * @param x the x
   * @return int int
   */
  public int reverse(int x) {
    // 已反转 & 待反转
    int rever = 0, left = x;
    while (left != 0) {
      // overflow
      if (rever < Integer.MIN_VALUE / 10 || rever > Integer.MAX_VALUE / 10) return 0;
      // tail digit
      rever = rever * 10 + left % 10;
      left /= 10;
    }
    return rever;
  }

  /**
   * 颠倒二进制位，反转二进制位，Java 则作为 unsigned 对待。
   *
   * <p>分治，两位互换 -> 4 -> 8 -> 16
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/reverse-bits/solution/dian-dao-er-jin-zhi-wei-by-leetcode-solu-yhxz/
   *
   * @param n the n
   * @return int int
   */
  public int reverseBits(int n) {
    // 每 2 位低 1 位和高 1 位交换，1010 是 a，0101 是 5
    n = n >>> 1 & 0x55555555 | (n & 0x55555555) << 1;
    // 每 4 位低 2 位和高 2 位交换，1100 是 c,0011 是 3
    n = n >>> 2 & 0x33333333 | (n & 0x33333333) << 2;
    // 每 8 位低 4 位和高 4 位交换
    n = n >>> 4 & 0x0f0f0f0f | (n & 0x0f0f0f0f) << 4;
    // 每 16 位低 8 位和高 8 位交换，1111 是 f
    n = n >>> 8 & 0x00ff00ff | (n & 0x00ff00ff) << 8;
    // 低 16 位与高 16 位交换
    return n >>> 16 | n << 16;
  }

  /**
   * 只出现一次的数字，II 参下，此 III，一个一次，其余三次。
   *
   * <p>使用一个 32 长的数组记录所有数值的每一位出现 1 的次数，再对数组的每一位进行 mod，重新拼凑出只出现一次的数
   *
   * <p>如果一个数字出现三次，那么它的二进制表示的每一位也出现三次。如果把所有出现三次的数字的二进制表示的每一位都分别加起来，那么每一位的和都能被 3 整除
   *
   * <p>如果某一位的和能被 3 整除,那么那个只出现一次的数字二进制表示中对应的那一位是 0 否则 1
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

  // 只出现一次的数字III，两个一次，其余两次。
  // TODO 参考
  // https://leetcode.cn/problems/single-number-iii/solution/cai-yong-fen-zhi-de-si-xiang-jiang-wen-ti-jiang-we/
  private int[] singleN2(int[] nums) {
    int xor = 0;
    for (int n : nums) {
      xor ^= n;
    }
    // 取异或值最后一个二进制位为 1 的数字作为 mask，如果是 1 则表示两个数字在这一位上不同。
    int mask = xor & (-xor);
    int[] res = new int[2];
    // 通过与这个 mask 进行与操作，如果为 0 的分为一个数组，为 1 的分为另一个数组。
    // 这样就把问题降低成：有一个数组每个数字都出现两次，有一个数字只出现了一次，求出该数字。
    // 对这两个子问题分别进行全异或就可以得到两个解。
    for (int n : nums) {
      if ((n & mask) == 0) res[0] ^= n;
      else res[1] ^= n;
    }
    return res;
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

  /**
   * 格雷编码
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/gray-code/solution/gray-code-jing-xiang-fan-she-fa-by-jyd/
   *
   * @param n the n
   * @return list list
   */
  public List<Integer> grayCode(int n) {
    List<Integer> res = new ArrayList<Integer>() {};
    res.add(0);
    int head = 1;
    for (int i = 0; i < n; i++) {
      for (int j = res.size() - 1; j >= 0; j--) {
        res.add(head + res.get(j));
      }
      head <<= 1;
    }
    return res;
  }
}

// LFU 参考图片即可
// https://leetcode-cn.com/problems/lfu-cache/solution/chao-xiang-xi-tu-jie-dong-tu-yan-shi-460-lfuhuan-c/
