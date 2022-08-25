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
   * Z字形变换
   *
   * <p>参考 https://leetcode.cn/problems/zigzag-conversion/solution/zzi-xing-bian-huan-by-jyd/
   *
   * @param s
   * @param numRows
   * @return
   */
  public String convert(String s, int numRows) {
    if (numRows < 2) return s;
    StringBuilder[] rows = new StringBuilder[numRows];
    for (int i = 0; i < numRows; i++) rows[i] = new StringBuilder();
    int r = 0, isForward = -1;
    for (char ch : s.toCharArray()) {
      rows[r].append(ch);
      if (r == 0 || r == numRows - 1) isForward *= -1;
      r += isForward;
    }
    StringBuilder res = new StringBuilder();
    for (StringBuilder i : rows) res.append(i);
    return res.toString();
  }

  /**
   * 加油站，求最大子序和的起始位
   *
   * <p>贪心，局部最优，当前累加和小于 0 则更新起始位置与累加和，并重新计算
   *
   * @param gas the gas
   * @param cost the cost
   * @return int int
   */
  public int canCompleteCircuit(int[] gas, int[] cost) {
    // sum 用于判断能否到达终点
    int preSum = 0, start = 0, sum = 0;
    for (int i = 0; i < gas.length; i++) {
      int n = gas[i] - cost[i]; // 当前站还剩下的，即当前元素的值
      sum += n;
      if (preSum + n > n) {
        preSum += n;
      } else {
        preSum = n;
        start = i;
      }
    }
    return sum < 0 ? -1 : start;
  }

  /**
   * 跳跃游戏，判断能否到达最后一个格，每格的数值表示可选的上界
   *
   * @param nums the nums
   * @return boolean boolean
   */
  public boolean canJump(int[] nums) {
    int maxIdx = 0;
    for (int i = 0; i <= maxIdx; i++) {
      maxIdx = Math.max(maxIdx, i + nums[i]);
      if (maxIdx >= nums.length - 1) return true;
    }
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
    int minCnt = 0, lo = 0, hi = 0;
    while (hi < nums.length - 1) {
      int maxIdx = 0;
      for (int i = lo; i <= hi; i++) maxIdx = Math.max(maxIdx, i + nums[i]);
      lo = hi + 1;
      hi = maxIdx;
      minCnt += 1;
    }
    return minCnt;
  }

  /**
   * 把数字翻译成字符串，返回方案数
   *
   * <p>以 xyzcba 为例，先取最后两位 即 ba，如果 ba>=26 必然不能分解成 f(xyzcb)+f(xyzc)，此时只能分解成 f(xyzcb)
   *
   * <p>但还有一种情况 ba<=9 即十位为 0 也不能分解
   *
   * <p>参考
   * https://leetcode.cn/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/solution/di-gui-qiu-jie-shuang-bai-by-xiang-shang-de-gua-ni/
   *
   * @param num
   * @return
   */
  public int translateNum(int num) {
    if (num <= 9) return 1;
    int ba = num % 100, tenPos = translateNum(num / 10);
    if (ba > 9 && ba < 26) return tenPos + translateNum(num / 100);
    return tenPos;
  }

  /**
   * 航班预订统计，贪心
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/corporate-flight-bookings/solution/5118_hang-ban-yu-ding-tong-ji-by-user9081a/
   *
   * @param bookings
   * @param n
   * @return
   */
  public int[] corpFlightBookings(int[][] bookings, int n) {
    int[] counter = new int[n];
    for (int[] b : bookings) {
      int lo = b[0], hi = b[1], w = b[2];
      counter[lo - 1] += w;
      if (hi < n) counter[hi] -= w;
    }
    for (int i = 1; i < n; i++) counter[i] += counter[i - 1];
    return counter;
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
    int[] cnts = new int[MaxScore + 1];
    for (int s : scores) cnts[s] += 1;
    for (int s = 1; s < MaxScore; s++) cnts[s] += cnts[s - 1];
    // 以学生维度
    int cnt = scores.length;
    double[] pcts = new double[cnt];
    for (int i = 0; i < cnt; i++) pcts[i] = 100.0 * cnts[scores[i]] / cnt;
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
   * LRU缓存机制 addToHead, moveToHead, removeTail, removeNode
   *
   * <p>hash 保证 O(1) 寻址 & 链表保证 DML 有序，双向保证将一个节点移到双向链表的头部，可以分成「删除该节点」&「在双向链表的头部添加节点」两步操作都 O(1)
   *
   * <p>扩展1，带超时，可以懒删除或 Daemon 随机 scan
   *
   * <p>扩展2，线程安全，空结点 throw exception，分别对 hash & 双向链表改用 ConcurrentHashMap & 读写锁，前者可以使用另一把锁代替
   *
   * <p>LFU 参考图片即可
   * https://leetcode-cn.com/problems/lfu-cache/solution/chao-xiang-xi-tu-jie-dong-tu-yan-shi-460-lfuhuan-c/
   */
  public class LRUCache {
    private final Map<Integer, DLinkedNode> k2n = new HashMap<>();
    private final DLinkedNode head = new DLinkedNode(), tail = new DLinkedNode(); // dummy
    private final int CAPACITY;
    /**
     * Instantiates a new Lru cache.
     *
     * @param capacity the capacity
     */
    public LRUCache(int capacity) {
      this.CAPACITY = capacity;
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
      DLinkedNode n = k2n.get(key);
      if (n == null) return -1;
      moveToHead(n);
      return n.value;
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
      DLinkedNode n = k2n.get(key);
      if (n != null) {
        n.value = value;
        moveToHead(n);
        return;
      }
      DLinkedNode newNode = new DLinkedNode(key, value);
      k2n.put(key, newNode);
      addToHead(newNode);
      if (k2n.size() > CAPACITY) {
        DLinkedNode tail = removeTail();
        k2n.remove(tail.key);
      }
    }

    private void addToHead(DLinkedNode n) {
      n.prev = head;
      n.next = head.next;
      head.next.prev = n;
      head.next = n;
    }

    // removeTail & addToHead
    private void moveToHead(DLinkedNode n) {
      removeNode(n);
      addToHead(n);
    }

    // revmoeNode
    private DLinkedNode removeTail() {
      DLinkedNode last = tail.prev;
      removeNode(last);
      return last;
    }

    private void removeNode(DLinkedNode n) {
      n.prev.next = n.next;
      n.next.prev = n.prev;
    }

    private class DLinkedNode {
      public int key, value;
      public DLinkedNode prev, next;

      public DLinkedNode() {}

      public DLinkedNode(int _key, int _value) {
        key = _key;
        value = _value;
      }
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
    private long min = Long.MAX_VALUE;
    private final Deque<Long> stack = new ArrayDeque<>();

    public void push(int n) {
      if (stack.isEmpty()) {
        stack.push(0L);
        min = n;
      } else {
        stack.push(n - min);
        min = Math.min(min, n);
      }
    }

    public void pop() {
      if (stack.isEmpty()) return;
      long pop = stack.pollLast();
      if (pop < 0) min -= pop;
    }

    public int top() {
      // 题设总是在非空栈上调用
      long top = stack.peekLast();
      return (int) (top < 0 ? min : top + min);
    }

    public int getMin() {
      return (int) min;
    }
  }

  /**
   * 用栈实现队列，双栈，in & out，均摊可以认为时间复制度为 O(1)
   *
   * <p>记忆，out & in & out & in
   */
  public class MyQueue {
    private final Deque<Integer> out = new ArrayDeque<>(), in = new ArrayDeque<>();

    public void push(int x) {
      in.offerLast(x);
    }

    public int pop() {
      int n = peek(); // 仅为复用
      out.pollLast();
      return n;
    }

    public int peek() {
      if (!out.isEmpty()) return out.peekLast();
      if (in.isEmpty()) return -1;
      while (!in.isEmpty()) out.offerLast(in.pollLast());
      return out.peekLast();
    }

    public boolean empty() {
      return out.isEmpty() && in.isEmpty();
    }
  }

  /**
   * 用队列实现栈 参考
   * https://leetcode.cn/problems/implement-stack-using-queues/solution/wu-tu-guan-fang-tui-jian-ti-jie-yong-dui-63d4/
   */
  public class MyStack {
    Queue<Integer> out = new LinkedList<>(), in = new LinkedList<>();

    public MyStack() {}

    public void push(int x) { // swap
      in.offer(x);
      while (!out.isEmpty()) in.offer(out.poll());
      Queue<Integer> tmp = out;
      out = in;
      in = tmp;
    }

    public int pop() {
      return out.poll();
    }

    public int top() {
      return out.peek();
    }

    public boolean empty() {
      return out.isEmpty();
    }
  }

  /**
   * 设计循环队列，三种实现方式，冗余一个元素/边界标记/计数器，此处选用冗余
   *
   * <p>front 指向队列头部，即首个有效数据的位置，而 rear 指向队尾下一个，即元素入队的位置
   *
   * <p>首尾碰撞为空，间隔一为满
   *
   * <p>取尾需要 (rear - 1 + CAPACITY) % CAPACITY
   *
   * <p>扩展1，并发安全，单个 push 多个 pop，采用 CAS 或参考 Disruptor
   */
  public class MyCircularQueue {
    private final int CAPACITY;
    private final int[] data;
    private int front, rear; // dummy head and tail 至少有一个位置不存放有效元素

    public MyCircularQueue(int k) {
      CAPACITY = k + 1;
      data = new int[CAPACITY];
    }

    public boolean isEmpty() {
      return front == rear;
    }

    public boolean isFull() {
      return (rear + 1) % CAPACITY == front;
    }

    public boolean enQueue(int value) {
      if (isFull()) return false;
      data[rear] = value;
      rear = (rear + 1) % CAPACITY;
      return true;
    }

    public boolean deQueue() {
      if (isEmpty()) return false;
      front = (front + 1) % CAPACITY;
      return true;
    }

    public int Front() {
      return isEmpty() ? -1 : data[front];
    }

    public int Rear() {
      return isEmpty() ? -1 : data[(rear - 1 + CAPACITY) % CAPACITY];
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

    public void insert(String word) {
      TireNode cur = root;
      for (char ch : word.toCharArray()) {
        if (cur.next[ch - 'a'] == null) cur.next[ch - 'a'] = new TireNode();
        cur = cur.next[ch - 'a'];
      }
      cur.isEnd = true;
    }

    public boolean search(String word) {
      TireNode target = lookup(word);
      return target == null ? false : target.isEnd;
    }

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
   * 设计哈希映射
   *
   * <p>扩展1，要求布隆过滤器
   */
  public class MyHashMap {
    private static final int CAPACITY = 769;
    private final List[] buckets;

    public MyHashMap() {
      buckets = new LinkedList[CAPACITY];
      for (int i = 0; i < CAPACITY; i++) {
        buckets[i] = new LinkedList<BucketNode>();
      }
    }

    private int hash(int key) {
      return key % CAPACITY;
    }

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
   * O(1) 时间插入、删除和获取随机元素
   *
   * <p>不能「使用拒绝采样」&「在数组非结尾位置添增删元素」，因此需要申请一个足够大的数组
   *
   * <p>以 val 为键，数组下标 loc 为值
   *
   * <p>参考 https://leetcode.cn/problems/insert-delete-getrandom-o1/solution/by-ac_oier-tpex/
   */
  public class RandomizedSet {
    private final int[] nums = new int[200010];
    private final Map<Integer, Integer> map = new HashMap<>();
    private int idx = -1;

    public boolean insert(int val) {
      if (map.containsKey(val)) return false;
      nums[idx] = val;
      map.put(val, idx);
      idx += 1;
      return true;
    }

    public boolean remove(int val) {
      if (!map.containsKey(val)) return false;
      int idxDeleted = map.remove(val);
      idx -= 1;
      nums[idxDeleted] = nums[idx];
      if (idxDeleted != idx) map.put(nums[idx], idxDeleted);
      return true;
    }

    public int getRandom() {
      return nums[new Random().nextInt(idx + 1)];
    }
  }

  /** 最大频率栈 */
  class FreqStack {
    private final Map<Integer, Integer> freq = new HashMap();
    private final Map<Integer, Deque<Integer>> group = new HashMap();
    private int maxfreq = 0;

    public void push(int x) {
      int f = freq.getOrDefault(x, 0) + 1;
      freq.put(x, f);
      if (f > maxfreq) maxfreq = f;
      group.computeIfAbsent(f, z -> new ArrayDeque<>()).push(x);
    }

    public int pop() {
      int x = group.get(maxfreq).pop();
      freq.put(x, freq.get(x) - 1);
      if (group.get(maxfreq).size() == 0) maxfreq -= 1;
      return x;
    }
  }
}

/** 计算相关 */
class MMath {
  /**
   * rand7生成rand10 即[1,10]，类似进制转换
   *
   * <p>参考
   * https://leetcode-cn.com/problems/implement-rand10-using-rand7/solution/cong-pao-ying-bi-kai-shi-xun-xu-jian-jin-ba-zhe-da/
   * https://www.cnblogs.com/ymjyqsx/p/9561443.html
   *
   * <p>数学推论 (randX-1)*Y+randY() 生成等概率 [1,X*Y]，只要 rand_N() 中 N 是 2 的倍数，就都可以用来实现 rand2()
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
      int n = (rand7() - 1) * 7 + rand7();
      // 拒绝采样，并返回 [1,10] 范围的随机数
      if (n <= 40) return n % 10 + 1;
    }
  }

  private int rand7() {
    return 0;
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
    if (x == 0 || x == 1) return x;
    int lo = 1, hi = x / 2;
    while (lo < hi) { // upper
      int mid = lo + (hi - lo + 1) / 2;
      if (mid <= x / mid) lo = mid;
      else hi = mid - 1;
    }
    return lo;
    //    if (x == 0) return 0;
    //    double n = x;
    //    while (true) {
    //      double cur = (n + x / n) * 0.5;
    //      // 后 n 位 1e^(-n)
    //      if (Math.abs(n - cur) < 1e-7) break;
    //      n = cur;
    //    }
    //    return (int) n;
  }

  /**
   * Pow(x,n)，快速幂
   *
   * @param x the x
   * @param n the n
   * @return double double
   */
  public double myPow(double x, int n) {
    //    long N = n; // 为测试通过
    return n < 0 ? 1.0 / quickMulti(x, -n) : quickMulti(x, n);
  }

  // 特判零次幂 & 递归二分 & 判断剩余幂
  private double quickMulti(double x, int n) {
    //    double res = 1.0, y = x;
    //    while (n > 0) {
    //      if (n % 2 == 1) res *= y;
    //      y *= cur;
    //      n /= 2;
    //    }
    //    return res;
    if (n == 0) return 1;
    double dou = Math.pow(quickMulti(x, n / 2), 2);
    return dou * (n % 2 == 0 ? 1 : x);
  }

  /**
   * 圆圈中最后剩下的数字，约瑟夫环，记住公式 res=(res+m)%i
   *
   * @param n the n
   * @param m the m
   * @return the int
   */
  public int lastRemaining(int n, int m) {
    int leftIdx = 0;
    for (int i = 2; i <= n; i++) leftIdx = (leftIdx + m) % i;
    return leftIdx;
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
    int candidate = nums[0], cnt = 0;
    for (int n : nums) {
      if (cnt == 0) {
        candidate = n;
        cnt = 1;
      } else if (n == candidate) {
        cnt += 1;
      } else {
        cnt -= 1;
      }
    }
    return candidate;
  }

  /**
   * 第N位数字/第N个数字
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
    int len = 1, base = 1;
    while (n > (long) 9 * len * base) {
      n -= 9 * len * base;
      len += 1;
      base *= 10;
    }
    int idx = n - 1, digit = idx % len;
    double num = Math.pow(10, len - 1) + idx / len;
    return (int) (num / Math.pow(10, len - digit - 1) % 10);
  }

  /**
   * 分数到小数，高精度除法，对比「两数相除」
   *
   * <p>每一位小数通过 *10 再除余计算，而循环小数可以通过判断被除数如 1/7 有没有出现过，及其结果的左边界来判断。
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/fraction-to-recurring-decimal/solution/pythonjavajavascript-gao-jing-du-chu-fa-44td5/
   *
   * @param num 分子
   * @param de 分母
   * @return
   */
  public String fractionToDecimal(int num, int de) {
    if (num * de < 0) return '-' + fractionToDecimal(Math.abs(num), Math.abs(de));
    StringBuilder res = new StringBuilder();
    res.append(num / de);
    res.append(".");
    Map<Long, Integer> de2LoIdx = new HashMap<>();
    for (long n = num % de; n != 0; n %= de) {
      // 出现过相同的余数说明开始出现循环小数
      if (de2LoIdx.containsKey(n)) {
        res.insert(de2LoIdx.get(n), "(");
        res.append(")");
        break;
      }
      // 记录每一个余数对应到结果中的位置
      de2LoIdx.put(n, res.length());
      // 模拟除法的过程
      n *= 10;
      res.append(n / de);
    }
    return res.toString();
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

  /**
   * 两数相除
   *
   * <p>实现一个「倍增乘法」，然后利用对于 x/y 必然落在区间 [0:x] 的规律进行二分
   *
   * <p>参考
   * https://leetcode.cn/problems/divide-two-integers/solution/shua-chuan-lc-er-fen-bei-zeng-cheng-fa-j-m73b/
   *
   * @param a
   * @param b
   * @return
   */
  public int divide(int a, int b) {
    boolean isNeg = false;
    if ((a > 0 && b < 0) || (a < 0 && b > 0)) isNeg = true;
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    long lo = 0, hi = a;
    while (lo < hi) {
      long mid = lo + (hi - lo + 1) / 2;
      if (mul(mid, b) <= a) lo = mid;
      else hi = mid - 1;
    }
    long n = isNeg ? -lo : lo;
    return n > Integer.MAX_VALUE || n < Integer.MIN_VALUE ? Integer.MAX_VALUE : (int) n;
  }

  private long mul(long a, long k) {
    long res = 0;
    while (k > 0) {
      if ((k & 1) == 1) res += a;
      k >>= 1; // 除以 2
      a += a; // 乘以 2
    }
    return res;
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
      if (cur == c) return true;
      if (cur > c) n2 -= 1;
    }
    return false;
  }

  /**
   * 1~n整数中1出现的次数/数字1的个数，如 1～12 这些整数中包含 1 的数字有 1、10、11、12 共 5 个
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
    int cnt = 0;
    int high = n / 10, cur = n % 10, low = 0, digit = 1;
    while (high > 0 || cur > 0) {
      // 状态
      cnt += high * digit;
      //      if (cur == 0) cnt += 0;
      if (cur == 1) cnt += low + 1;
      if (cur > 1) cnt += digit;
      // 递推
      low += cur * digit;
      cur = high % 10;
      high /= 10;
      digit *= 10;
    }
    return cnt;
  }
}

/**
 * 收集图相关，参考 https://oi-wiki.org/graph，有如下题型
 *
 * <p>环路 & 两点连通性
 *
 * <p>最短路
 *
 * <p>路径总数
 *
 * <p>求两点之间的权值最小的路径
 */
class GGraph {
  /**
   * 课程表/检测循环依赖，返回是否 DAG，拓扑排序/Kahn 算法
   *
   * <p>拓扑排序有两种模式，从入度思考，即从前往后排序，采用 BFS，入度为 0 的节点在拓扑排序中一定排在前面, 然后删除和该节点对应的边, 迭代寻找入度为 0 的节点
   *
   * <p>从出度思考，即从后往前排序，采用 DFS，出度为 0 的节点在拓扑排序中一定排在后面, 然后删除和该节点对应的边, 迭代寻找出度为 0 的节点。
   *
   * <p>参考 https://mp.weixin.qq.com/s/pCRscwKqQdYYN7M1Sia7xA
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return boolean boolean
   */
  public boolean canFinish(int V, int[][] prerequisites) {
    int[] indegrees = new int[V];
    List<List<Integer>> table = buildTable(prerequisites, V, indegrees);
    Queue<Integer> queue = new LinkedList();
    for (int i = 0; i < V; i++) if (indegrees[i] == 0) queue.offer(i);
    while (!queue.isEmpty()) {
      V -= 1;
      for (int adj : table.get(queue.poll())) {
        indegrees[adj] -= 1;
        if (indegrees[adj] == 0) queue.offer(adj);
      }
    }
    return V == 0;
  }

  private List<List<Integer>> buildTable(int[][] prerequisites, int V, int[] indegrees) {
    List<List<Integer>> table = new ArrayList(V);
    for (int i = 0; i < V; i++) table.add(new LinkedList());
    for (int[] cp : prerequisites) {
      int from = cp[0], to = cp[1];
      indegrees[to] += 1;
      table.get(from).add(to);
    }
    return table;
  }

  /**
   * 课程表II，比 I 新增记录即可
   *
   * <p>若存在循环依赖则返回空，否则返回可行的编译顺序
   *
   * <p>参考 https://mp.weixin.qq.com/s/pCRscwKqQdYYN7M1Sia7xA
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return int [ ]
   */
  public int[] findOrder(int V, int[][] prerequisites) {
    int[] indegrees = new int[V];
    List<List<Integer>> table = buildTable(prerequisites, V, indegrees);
    Queue<Integer> queue = new LinkedList();
    for (int i = 0; i < V; i++) if (indegrees[i] == 0) queue.offer(i);
    int[] paths = new int[V];
    while (!queue.isEmpty()) {
      int ver = queue.poll();
      V -= 1;
      paths[V] = ver;
      for (int adj : table.get(ver)) {
        indegrees[adj] -= 1;
        if (indegrees[adj] == 0) queue.offer(adj);
      }
    }
    return V == 0 ? paths : new int[0];
  }

  /**
   * 网络延迟时间，到达所有结点的最短路径，Dijkstra
   *
   * <p>参考
   * https://leetcode.cn/problems/network-delay-time/solution/gong-shui-san-xie-yi-ti-wu-jie-wu-chong-oghpz/
   *
   * @param times the times
   * @param n the n
   * @param k the k
   * @return int int
   */
  public int networkDelayTime(int[][] ts, int n, int k) {
    int V = 110, E = 6010;
    // 邻接表链式前向星存图，head[i] 第 i 节点对应边的集合的头节点
    int[] heads = new int[V];
    // to[i] 第 i 边指向的节点，若无向图则需占用两项
    // weights[i] 第 i 边权重
    // next[i] 第 i 边的起点存储的某个邻接节点，由于是以链表的形式进行存边，该数组就是用于找到下一条边
    int[] tos = new int[E], weights = new int[E], nexts = new int[E];
    buildTable(ts, heads, tos, weights, nexts);
    // 遍历最短路
    int maxDist = 0;
    int[] dists = dijkstra(V, k, heads, tos, weights, nexts);
    for (int d : dists) maxDist = Math.max(maxDist, d);
    return maxDist > Integer.MAX_VALUE / 2 ? -1 : maxDist;
  }

  private void buildTable(int[][] ts, int[] heads, int[] tos, int[] weights, int[] nexts) {
    Arrays.fill(heads, -1); // 初始化链表头
    int idx = 0;
    for (int[] t : ts) {
      int from = t[0], to = t[1], w = t[2];
      tos[idx] = to;
      nexts[idx] = heads[from];
      weights[idx] = w;
      heads[from] = idx;
      idx += 1;
    }
  }

  private int[] dijkstra(int V, int k, int[] heads, int[] tos, int[] weights, int[] nexts) {
    // dist[x] = y 代表从「源点/起点」到 x 的最短距离为 y
    int[] dists = new int[V];
    Arrays.fill(dists, Integer.MAX_VALUE);
    dists[k] = 0; // 只有起点最短距离为 0
    dists[0] = 0; // 题设 ID 始于 1
    // 小根堆存储所有可用于更新的点，以 (点编号, 到起点的距离) 进行存储，优先弹出「最短距离」较小的点
    PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
    pq.offer(new int[] {k, 0});
    // 记录已经被更新过
    boolean[] visited = new boolean[V];
    while (!pq.isEmpty()) {
      int v = pq.poll()[0];
      if (visited[v]) continue;
      // 标记该点「已更新」，并使用该点更新其他点的「最短距离」
      visited[v] = true;
      for (int i = heads[v]; i != -1; i = nexts[i]) {
        int adj = tos[i], d = dists[v] + weights[i];
        if (dists[adj] <= d) continue;
        dists[adj] = d;
        pq.offer(new int[] {adj, dists[adj]});
      }
    }
    return dists;
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
    int V = graph.length, MASK = 1 << V, INF = Integer.MAX_VALUE;
    // Floyd 求两点的最短路径
    int[][] dist = new int[V][V];
    for (int i = 0; i < V; i++) {
      for (int j = 0; j < V; j++) dist[i][j] = INF;
      for (int j : graph[i]) dist[i][j] = 1;
    }
    for (int k = 0; k < V; k++) {
      for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
          dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
        }
      }
    }
    // DP 过程，如果从 i 能够到 j 的话，使用 i 到 j 的最短距离（步长）来转移
    int[][] f = new int[MASK][V];
    // 起始时，让所有状态的最短距离（步长）为正无穷
    for (int i = 0; i < MASK; i++) Arrays.fill(f[i], INF);
    // 由于可以将任意点作为起点出发，可以将这些起点的最短距离（步长）设置为 0
    for (int i = 0; i < V; i++) f[1 << i][i] = 0;
    // 枚举所有的 state
    for (int s = 0; s < MASK; s++) {
      // 枚举 state 中已经被访问过的点
      for (int i = 0; i < V; i++) {
        if (((s >> i) & 1) == 0) continue;
        // 枚举 state 中尚未被访问过的点
        for (int j = 0; j < V; j++) {
          if (((s >> j) & 1) == 1) continue;
          f[s | (1 << j)][j] = Math.min(f[s | (1 << j)][j], f[s][i] + dist[i][j]);
        }
      }
    }
    int minLen = INF;
    for (int i = 0; i < V; i++) minLen = Math.min(minLen, f[MASK - 1][i]);
    return minLen;
  }

  /**
   * 找到最终的安全状态，不在环内的节点，三色标记/拓扑排序
   *
   * @param graph
   * @return
   */
  public List<Integer> eventualSafeNodes(int[][] graph) {
    int V = graph.length;
    int[] recStack = new int[V];
    List<Integer> vtxs = new ArrayList<>();
    for (int vtx = 0; vtx < V; vtx++) {
      if (dfs20(graph, recStack, vtx)) vtxs.add(vtx);
    }
    return vtxs;
  }

  private final int WHILE = 0, GREY = 1, BLACK = 2;

  private boolean dfs20(int[][] graph, int[] recStack, int v) {
    if (recStack[v] != WHILE) return recStack[v] == BLACK;
    recStack[v] = GREY;
    for (int adj : graph[v]) { // 题设邻接列表
      if (!dfs20(graph, recStack, adj)) return false;
    }
    recStack[v] = BLACK;
    return true;
  }

  private int diameter = 0; // 「N叉树的直径」结果

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
      int neighbor = edge.to;
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

  // 「克隆图」
  private final Map<Node, Node> recStack = new HashMap<>();

  /**
   * 克隆图，deep clone undirected connected graph，DFS
   *
   * @param node
   * @return
   */
  public Node cloneGraph(Node node) {
    if (node == null) return node;
    if (recStack.containsKey(node)) return recStack.get(node);
    // 为深拷贝我们不会克隆它的邻居的列表
    Node cloneNode = new Node(node.val, new ArrayList());
    recStack.put(node, cloneNode);
    // 遍历该节点的邻居并更新克隆节点的邻居列表
    for (Node nbh : node.neighbors) cloneNode.neighbors.add(cloneGraph(nbh));
    return cloneNode;
  }

  private class Edge {

    // 链式前向星存图实现邻接表，方便起见通常拆分为三个数组
    int to, weight, next;

    /**
     * Instantiates a new Edge.
     *
     * @param to the end
     * @param w the w
     */
    Edge(int to, int w) {
      this.to = to;
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
    int rever = 0, left = x; // 已反转 & 待反转
    while (left != 0) {
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
   * 只出现一次的数字，此 III，II 参下，一个一次，其余三次。
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
  //  public int singleNumber(int[] nums) {
  //    int xor = 0;
  //    for (int n : nums) xor ^= n;
  //    int mask = xor & (-xor);
  //    int[] ans = new int[2];
  //    for (int n : nums) {
  //      if ((n & mask) == 0) ans[0] ^= n;
  //      else ans[1] ^= n;
  //    }
  //    return ans;
  //  }

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
