package com.zh1095.demo.improved.algorithmn;

import java.util.*;

interface MountainArray {
  default int get(int index) {
    return 0;
  }

  default int length() {
    return 0;
  }
}

/**
 * 收集数组相关，包括如下类型，部分查找不到则至 DDP
 *
 * <p>排序，掌握快速 & 归并 & 堆即可，参考
 * https://leetcode-cn.com/problems/sort-an-array/solution/fu-xi-ji-chu-pai-xu-suan-fa-java-by-liweiwei1419/
 *
 * <p>区分 start & end 与 lo & hi，前组静态，表示索引或上下界，后组动态，大部分情况向中碰撞
 *
 * <p>随机遍历则 for (int i = 0; i < nums.length; i++) & continue 即可，否则
 *
 * <p>for (List<Integer> level : triangle) { for (int num : level) {} } or triangle.forEach(level ->
 * { level.forEach(num -> {}); });
 *
 * <p>s.charAt(i) 会检查字符串的下标是否越界，因此非随机遍历直接 s.toCharArray() 即可
 *
 * <p>array to list Arrays.stream(nums).boxed().collect(Collectors.toList());
 *
 * <p>list to array list.stream().mapToInt(i -> i).toArray();
 *
 * @author cenghui
 */
public class AArray extends DefaultArray {
  /**
   * 二分查找，下方 FFind 统一该写法，参考 https://www.zhihu.com/question/36132386
   *
   * <p>思想是基于比较点，不断剔除「不合格区域」，但不能保证「最后幸存的 l=r 区域是合格的」，因此最后应判定 l 的值是否正确
   *
   * <p>扩展1，重复，改为 nextIdx
   *
   * <p>扩展2，参下「在排序数组中查找元素的第一个和最后一个位置」
   *
   * @param nums the nums
   * @param target the target
   * @return int int
   */
  public int search(int[] nums, int target) {
    // 特判，避免当 target 小于 nums[0] & nums[end] 时多次循环运算
    if (target < nums[0] || target > nums[nums.length - 1]) return -1;
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (nums[mid] < target) lo = mid + 1;
      else if (nums[mid] == target) return mid;
      else if (nums[mid] > target) hi = mid;
    }
    return nums[lo] == target ? lo : -1;
  }

  /**
   * 合并两个有序数组，题设不需要滤重，逆向，参考合并两个有序链表
   *
   * <p>扩展1，滤重，取代为 nextIdx
   *
   * @param nums1 the nums 1
   * @param m the m
   * @param nums2 the nums 2
   * @param n the n
   * @return
   */
  public void merge(int[] nums1, int m, int[] nums2, int n) {
    int p1 = m - 1, p2 = n - 1;
    for (int i = m + n - 1; i >= 0; i--) {
      int bigger,
          n1 = p1 < 0 || p1 > nums1.length - 1 ? Integer.MIN_VALUE : nums1[p1],
          n2 = p2 < 0 || p2 > nums2.length - 1 ? Integer.MIN_VALUE : nums2[p2];
      if (n1 <= n2) {
        bigger = n1;
        p1 -= 1; // nextIdx()
      } else {
        bigger = n2;
        p2 -= 1; // nextIdx()
      }
      nums1[i] = bigger;
    }
  }

  /**
   * 有序数组的平方，入参已升序
   *
   * @param nums
   * @return
   */
  public int[] sortedSquares(int[] nums) {
    int len = nums.length, lo = 0, hi = len - 1;
    int[] res = new int[len];
    for (int i = len - 1; i > -1; i--) {
      int a = nums[lo] * nums[lo], b = nums[hi] * nums[hi];
      if (a > b) {
        res[i] = a;
        lo += 1;
      } else {
        res[i] = b;
        hi -= 1;
      }
    }
    return res;
  }

  /**
   * 两个数组的交集，重复 & 顺序
   *
   * <p>扩展1，有序则双指针 & 二分，否则对较小的数组建立映射再遍历较大者
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return int [ ]
   */
  public int[] intersection(int[] nums1, int[] nums2) {
    Arrays.sort(nums1);
    Arrays.sort(nums2);
    int l1 = nums1.length, l2 = nums2.length;
    int[] res = new int[l1 + l2];
    int cur = 0, p1 = 0, p2 = 0;
    while (p1 < l1 && p2 < l2) {
      int n1 = nums1[p1], n2 = nums2[p2];
      if (n1 < n2) {
        p1 += 1;
      } else if (n1 > n2) {
        p2 += 1;
      } else if (n1 == n2) {
        // 保证加入元素的唯一性
        if (cur == 0 || n1 != res[cur - 1]) {
          res[cur] = n1;
          cur += 1;
        }
        p1 += 1;
        p2 += 1;
      }
    }
    return Arrays.copyOfRange(res, 0, cur);
  }

  /**
   * 获取最大与第二大的数，无序数组
   *
   * @param nums
   * @return
   */
  public int[] getMaxAndSecond(int[] nums) {
    if (nums.length < 1) return new int[2];
    int max = nums[0], second = Integer.MIN_VALUE;
    for (int i = 1; i < nums.length; i++) {
      if (nums[i] <= max) {
        second = Math.max(second, nums[i]);
        continue;
      }
      second = max;
      max = nums[i];
    }
    return new int[] {max, second};
  }

  /**
   * 扑克牌中的顺子，A1，J11，Q12，K13，而大小王 0 ，可以看成任意数字，相当于判断无序数组是否完全连续，两类思路
   *
   * <p>Set后遍历
   *
   * <p>排序后遍历
   *
   * @param nums the nums
   * @return boolean boolean
   */
  public boolean isStraight(int[] nums) {
    Arrays.sort(nums);
    int joker = 0;
    for (int i = 0; i < 4; i++) {
      if (nums[i] == 0) joker += 1; // 统计大小王数量
      else if (nums[i] == nums[i + 1]) return false; // 若有重复，提前返回 false
    }
    return nums[4] - nums[joker] < 5; // 最大牌 - 最小牌 < 5 则可构成顺子
  }

  /**
   * 打乱数组，Shuffle 洗牌算法即可
   *
   * @author cenghui
   */
  public class Solution {
    private final int[] nums;
    private final Random random = new Random();

    /**
     * Instantiates a new Solution.
     *
     * @param _nums the nums
     */
    public Solution(int[] _nums) {
      nums = _nums;
    }

    /**
     * Reset int [ ].
     *
     * @return the int [ ]
     */
    public int[] reset() {
      return nums;
    }

    /**
     * [i,n) 随机抽取一个下标 j & 将第 i 个元素与第 j 个元素交换
     *
     * @return int [ ]
     */
    public int[] shuffle() {
      int[] res = nums.clone();
      for (int i = 0; i < nums.length; i++) {
        int randomIdx = random.nextInt(nums.length - i);
        swap(res, i, i + randomIdx);
      }
      return res;
    }
  }
}

/**
 * 基于比较的排序的平均时间复杂度均是 nlogn，快排的推导参考 https://nathanli.tech/2022/03/03/QuickSort/
 *
 * <p>数组全排列共有 n! 种情况，而二分每次最多能排除一半的情况，根据斯特林级数，算法的渐进复杂度为 O(log(n!)) = O(nlogn)
 *
 * <p>最坏时间复杂度出现在每次划分均为 n-1 & 0，为 n^2
 *
 * <p>最好时间复杂度出现在每次划分均为 n/2 & n/2-1，为 nlogn
 *
 * <p>平均时间复杂度出现在上述二者的分布为均分时，为 nlogn
 *
 * <p>链表快排参考「排序链表」
 */
class QQuick extends DefaultArray {
  private final Random random = new Random();

  /**
   * 快速排序，三路，循环不变量
   *
   * <p>选哨 & 虚拟头尾 & 遍历
   *
   * @param nums the nums
   * @param lo the lo
   * @param hi the hi
   */
  public void quickSort(int[] nums, int lo, int hi) {
    if (lo >= hi) return; // 对 [lo:hi] 快排
    int pivotIdx = lo + random.nextInt(hi - lo + 1), pivot = nums[pivotIdx];
    // 下方需要确保虚拟头 <pivot，因此从 lt+1 开始遍历
    swap(nums, pivotIdx, lo);
    // 虚拟头尾，保证界外，因此下方需要先步进，再 swap
    int lt = lo, cur = lt + 1, gt = hi + 1;
    while (cur < gt) {
      if (nums[cur] < pivot) {
        lt += 1;
        swap(nums, cur, lt);
        cur += 1;
      } else if (nums[cur] == pivot) {
        cur += 1;
      } else if (nums[cur] > pivot) {
        gt -= 1;
        swap(nums, cur, gt);
      }
    }
    // 扰动，保证等概率分布
    swap(nums, lo, lt);
    quickSort(nums, lo, lt - 1);
    quickSort(nums, gt, hi);
  }

  /**
   * 颜色分类，三路快排即可
   *
   * @param nums the nums
   */
  public void sortColors(int[] nums) {
    int pivot = 1;
    // 虚拟头尾，保证界外
    int lt = -1, cur = 0, gt = nums.length;
    while (cur < gt) {
      if (nums[cur] < pivot) {
        lt += 1;
        swap(nums, cur, lt);
        cur += 1;
      } else if (nums[cur] == pivot) {
        cur += 1;
      } else if (nums[cur] > pivot) {
        gt -= 1;
        swap(nums, cur, gt);
      }
    }
  }
}

/** 堆排序相关，建堆 */
class HHeap extends DefaultArray {
  /**
   * 数组中的第k个最大元素，原地维护小根堆
   *
   * <p>堆化 [0,k] & 依次入堆 [k+1,l-1] 的元素 & 最终堆顶即 [0]，具体过程参考动图
   * https://leetcode.cn/problems/sort-an-array/solution/pai-xu-shu-zu-by-leetcode-solution/
   *
   * <p>扩展1，寻找两个有序数组的第 k 大，参下「寻找两个有序数组的中位数」
   *
   * <p>TODO 扩展2，判断 num 是否为第 k 大，有重复，partition 如果在 K 位置的左边和右边都遇到该数，直接结束，否则直到找到第 k 大的元素比较是否为同一个数
   *
   * <p>扩展3，如何只选出 [n, m]，分别建两个长度为 n & m-n 的小根堆，优先入前者，前者出队至入后者，后者不允则舍弃
   *
   * <p>扩展4，不同量级的策略，如 10 & 100 & 10k 分别为计数排序，快速选择 or 建堆，分治 & 外排
   *
   * @param nums the nums
   * @param k the k
   * @return the int
   */
  public int findKthLargest(int[] nums, int k) {
    // 对前 k 个元素建小根堆，即堆化 [0,k-1] 区间的元素
    //    for (int i = 0; i < k; i++) {
    //      swim(nums, i);
    //    }
    heapify(nums, k - 1);
    // 剩下的元素与堆顶比较，若大于堆顶则去掉堆顶，再将其插入
    for (int i = k; i < nums.length; i++) {
      if (priorityThan(nums[i], nums[0])) continue;
      swap(nums, 0, i);
      sink(nums, 0, k - 1);
    }
    return nums[0];
  }

  /**
   * 堆是具有以下性质的完全二叉树，每个结点的值都大于或等于其左右孩子结点的值，称为大顶堆，反之，小顶堆
   *
   * <p>切换 priorityRThan 为 > 即可
   *
   * @param nums the nums
   */
  public void heapSort(int[] nums) {
    int end = nums.length - 1;
    heapify(nums, end);
    for (int i = end; i >= 1; i--) { // 循环不变量，[0:idx] 堆有序
      swap(nums, i, 0); // 把堆顶元素交换到数组末尾
      sink(nums, 0, i - 1); // 下标 0 位置下沉操作，使得 [0:i] 堆有序
    }
  }

  // 从首个叶结点开始逐层下沉，即堆化 nums[0:end]
  private void heapify(int[] nums, int end) {
    for (int i = end / 2; i > -1; i--) {
      sink(nums, i, end);
    }
  }

  // 从下到上调整堆，分别取左右子结点 2*cur+1 与 +2 判断
  private void sink(int[] heap, int start, int end) {
    int cur = start, child = 2 * cur + 1;
    while (2 * cur + 1 <= end) {
      if (child + 1 <= end && priorityThan(heap[child + 1], heap[child])) {
        child += 1;
      }
      if (priorityThan(heap[cur], heap[child])) break;
      swap(heap, cur, child);
      cur = child;
      child = 2 * cur + 1;
    }
  }

  // 从下到上调整堆，取父结点 (cur-1)/2
  private void swim(int[] heap, int idx) {
    int cur = idx, parent = (cur - 1) / 2;
    while (cur > 0 && priorityThan(heap[cur], heap[parent])) {
      swap(heap, cur, parent);
      cur = parent;
      parent = (cur - 1) / 2;
    }
  }

  // v1 的优先级是否高于 v2 此处小根堆
  private boolean priorityThan(int v1, int v2) {
    return v1 < v2;
  }

  /**
   * 字符串出现次数 topk，小根堆 nlogk
   *
   * <p>先按次数排，相同则按字典序排
   *
   * @param strings string字符串一维数组 strings
   * @param k int整型 the k
   * @return string字符串二维数组
   */
  public String[][] topKstrings(String[] strings, int k) {
    String[][] res = new String[k][2];
    Map<String, Integer> counter = new HashMap<>();
    for (String str : strings) {
      counter.put(str, counter.getOrDefault(str, 0) + 1);
    }
    PriorityQueue<String[]> pq =
        new PriorityQueue<>(
            k + 1,
            (o1, o2) -> {
              return Integer.parseInt(o1[1]) == Integer.parseInt(o2[1])
                  ? o2[0].compareTo(o1[0])
                  : Integer.parseInt(o1[1]) - Integer.parseInt(o2[1]);
            });
    for (String str : counter.keySet()) {
      pq.offer(new String[] {str, counter.get(str).toString()});
      if (pq.size() > k) pq.poll();
    }
    int idx = k - 1;
    while (!pq.isEmpty()) {
      res[idx] = pq.poll();
      idx -= 1;
    }
    return res;
  }

  /**
   * 最接近原点的k个点，大根堆排序
   *
   * @param points
   * @param K
   * @return
   */
  public int[][] kClosest(int[][] points, int K) {
    int[][] nodes = new int[K][2];
    PriorityQueue<int[]> pq =
        new PriorityQueue<>(
            K, (p1, p2) -> p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1]);
    for (int[] p : points) {
      if (pq.size() < K) {
        pq.offer(p);
        continue;
      }
      // 判断当前点的距离是否小于堆中的最大距离
      if (pq.comparator().compare(p, pq.peek()) > 0) {
        pq.poll();
        pq.offer(p);
      }
    }
    for (int i = 0; i < K; i++) {
      nodes[i] = pq.poll();
    }
    return nodes;
  }

  /**
   * 会议室II，返回最少重叠数，抽象成「上下车」问题。
   *
   * <p>满足最繁忙的时间点即可，因此区间有交集则暂存，否则移除末端最小的，因此使用小根堆。
   *
   * <p>参考 https://wyh317.github.io/2021/02/17/252.%E4%BC%9A%E8%AE%AE%E5%AE%A4/
   *
   * @param meetings
   * @return
   */
  public int minMeetingRooms(int[][] intervals) {
    Arrays.sort(intervals, (v1, v2) -> (v1[0] - v2[0]));
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    int count = 0;
    for (int[] itv : intervals) {
      while (!minHeap.isEmpty() && itv[0] >= minHeap.peek()) {
        minHeap.poll();
      }
      minHeap.add(itv[1]);
      count = Math.max(count, minHeap.size());
    }
    return count;
  }

  /**
   * 数据流的中位数/多个无序数组的中位数，分别使用两个堆，保证大根堆元素始终多一个
   *
   * <p>参考
   * https://leetcode-cn.com/problems/find-median-from-data-stream/solution/gong-shui-san-xie-jing-dian-shu-ju-jie-g-pqy8/
   *
   * <p>扩展1，所有整数都在 [1,100]，空间上优化，额外建立一个长度为 101
   * 的桶统计每个数的出现次数，同时记录数据流中总的元素数量，每次查找中位数时，先计算出中位数是第几位，从前往后扫描所有的桶得到答案
   *
   * <p>扩展2，PCT99 整数都在 [1,100]，对于 1% 采用哨兵机制进行解决，在常规的最小桶和最大桶两侧分别维护一个有序序列，即建立一个代表负无穷和正无穷的桶
   */
  public class MedianFinder {
    private final PriorityQueue<Integer> minHeap = new PriorityQueue<>((a, b) -> a - b),
        maxHeap = new PriorityQueue<>((a, b) -> b - a);

    public void addNum(int num) {
      if (maxHeap.size() == minHeap.size()) {
        if (!minHeap.isEmpty() && num > minHeap.peek()) {
          maxHeap.add(minHeap.poll());
          minHeap.add(num);
        } else {
          maxHeap.add(num);
        }
      } else {
        if (maxHeap.peek() > num) {
          minHeap.add(maxHeap.poll());
          maxHeap.add(num);
        } else {
          minHeap.add(num);
        }
      }
    }

    public double findMedian() {
      return maxHeap.size() == minHeap.size()
          ? (maxHeap.peek() + minHeap.peek()) / 2.0
          : maxHeap.peek();
    }
  }

  /** 数据流的第k大，维护一个小根堆 */
  public class KthLargest {
    private final int[] minheap;
    private final int ranking;
    int size = 0;

    public KthLargest(int k, int[] nums) {
      minheap = new int[k];
      ranking = k;
      for (int i = 0; i < nums.length; i++) {
        add(nums[i]);
      }
    }

    public int add(int val) {
      if (size < ranking) {
        minheap[size] = val;
        swim(minheap, size);
        size += 1;
      } else if (minheap[0] < val) {
        minheap[0] = val;
        sink(minheap, 0, ranking - 1);
      }
      return minheap[0];
    }
  }
}

class MMerge extends DefaultArray {
  private int count = 0; // 「数组中的逆序对」

  /**
   * 归并排序，up-to-bottom 递归，先分后合
   *
   * <p>bottom-to-up 迭代，参考排序链表
   *
   * @param nums the nums
   * @param start the lo
   * @param hi the hi
   */
  public void mergeSort(int[] nums, int start, int end) {
    divideAndCount(nums, new int[nums.length], 0, nums.length - 1);
  }

  /**
   * 数组中的逆序对，「归并排序」基本一致
   *
   * @param nums the nums
   * @return int int
   */
  public int reversePairs(int[] nums) {
    divideAndCount(nums, new int[nums.length], 0, nums.length - 1);
    return count;
  }

  // 合并 nums[lo:mid] & nums[mid+1:hi] 即排序区间 [lo,hi]
  // 四种情况，其一遍历结束 & 比较
  // 写成 < 会丢失稳定性，因为相同元素原来靠前的排序以后依然靠前，因此排序稳定性的保证必需 <=
  private void divideAndCount(int[] nums, int[] tmp, int start, int end) {
    // 对于双指针，假如迭代内部基于比较，则不需要=，假如需要统计每个元素，则需要
    if (start >= end) return;
    int mid = start + (end - start) / 2;
    divideAndCount(nums, tmp, start, mid);
    divideAndCount(nums, tmp, mid + 1, end);
    // curing 因为此时 [lo,mid] & [mid+1,hi] 分别有序，否则说明二者在数轴上范围存在重叠
    if (nums[mid] <= nums[mid + 1]) return;
    // 合并 nums[start,mid] & nums[mid+1,end]
    count += mergeAndCount(nums, tmp, start, mid, end);
  }

  private int mergeAndCount(int[] nums, int[] tmp, int start, int mid, int end) {
    if (end > start) {
      System.arraycopy(nums, start, tmp, start, end - start + 1);
    }
    int count = 0, p1 = start, p2 = mid + 1;
    for (int i = start; i <= end; i++) {
      if (p1 == mid + 1) {
        nums[i] = tmp[p2];
        p2 += 1;
      } else if (p2 == end + 1) {
        nums[i] = tmp[p1];
        p1 += 1;
      } else if (tmp[p1] <= tmp[p2]) {
        nums[i] = tmp[p1];
        p1 += 1;
      } else if (tmp[p1] > tmp[p2]) {
        nums[i] = tmp[p2];
        p2 += 1;
        count += mid - p1 + 1; // 统计逆序的数目
      }
    }
    return count;
  }

  /**
   * 合并区间，逐一比较上一段尾 & 当前段首
   *
   * @param intervals
   * @return
   */
  public int[][] merge(int[][] intervals) {
    int[][] res = new int[intervals.length][2];
    // 按照区间起始位置排序
    Arrays.sort(intervals, (v1, v2) -> v1[0] - v2[0]);
    int idx = -1;
    for (int[] interval : intervals) {
      if (idx == -1 || interval[0] > res[idx][1]) {
        idx += 1;
        res[idx] = interval;
      } else {
        res[idx][1] = Math.max(res[idx][1], interval[1]);
      }
    }
    return Arrays.copyOf(res, idx + 1);
  }

  /**
   * 会议室，判断是否有交集即可，即某个会议开始时，上一个会议是否结束。
   *
   * <p>「会议室II」参上
   *
   * @param intervals
   * @return
   */
  public boolean canAttendMeetings(int[][] intervals) {
    Arrays.sort(intervals, (v1, v2) -> (v1[0] - v2[0]));
    for (int i = 1; i < intervals.length; i++) {
      if (intervals[i][0] < intervals[i - 1][1]) return false;
    }
    return true;
  }

  /**
   * 螺丝螺母匹配，快速排序 & 分治
   *
   * <p>TODO 参考
   *
   * <p>设 Si->Nj
   *
   * @param screws
   * @param nuts
   * @return
   */
  public int[][] aggrateScrew2Nut(int[] screws, int[] nuts) {
    int len = screws.length;
    int[][] screw2Nut = new int[len][len]; // 可能存在不成对。

    for (int screw : screws) {}

    return screw2Nut;
  }

  // 可调用函数
  //  private int match(int screw, int nut) {}
}

/**
 * 二分，参考上方 AArray.search 的写法
 *
 * <p>lo<=hi 明确碰撞的含义
 */
class DichotomyClassic extends DefaultArray {
  /**
   * 寻找两个有序数组的中位数，数组内去重，数组间重叠，根据排位二分，掌握迭代即可
   *
   * <p>topK 需要去重，而中位则保留
   *
   * <p>对于两个有序数组，逆序则顺序遍历求第 k 大，正序则顺序遍历求第 k 小，反之，二者均需要逆序遍历
   *
   * <p>对于 123 & 334 的中位数和 top3 分别是 3 & 2
   *
   * <p>扩展1，求两个逆序数组 topK，单个数组不包含重复则复用即可
   *
   * <p>扩展2，无序数组找中位数，建小根堆 len/2+1 奇数则堆顶，否则出队一次 & 堆顶取平均，海量数据参考「数据流的中位数」
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return double double
   */
  public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int l1 = nums1.length, l2 = nums2.length;
    int ranking = (l1 + l2 + 1) / 2, rankingMore = (l1 + l2 + 2) / 2;
    // 将偶数和奇数的情况合并，奇数则求两次同样的 k
    //    return (getkSmallElement(nums1, 0, n - 1, nums2, 0, m - 1, ranking)
    //            + getkSmallElement(nums1, 0, n - 1, nums2, 0, m - 1, rankingMore))
    //        * 0.5;
    return (getLargestElement(nums1, nums2, ranking) + getLargestElement(nums1, nums2, rankingMore))
        * 0.5;
  }

  // 扩展1，求第 k 小参下 annotate
  private int getLargestElement(int[] nums1, int[] nums2, int k) {
    int l1 = nums1.length, l2 = nums2.length;
    int p1 = l1 - 1, p2 = l2 - 1, ranking = k;
    //    int p1 = 0, p2 = 0, ranking = k;
    while (p1 > -1 && p2 > -1 && ranking > 1) {
      //    while (p1 < l1 && p2 < l2 && ranking > 1) {
      int half = ranking / 2,
          newP1 = Math.max(p1 - half, -1) + 1,
          newP2 = Math.max(p2 - half, -1) + 1;
      //          newP1 = Math.min(p1 + half, l1) - 1,
      //          newP2 = Math.min(p2 + half, l2) - 1;
      int n1 = nums1[newP1], n2 = nums2[newP2];
      if (n1 >= n2) {
        ranking -= (p1 - newP1 + 1);
        p1 = newP1 - 1;
      } else {
        ranking -= (p2 - newP2 + 1);
        p2 = newP2 - 1;
      }
      //      if (n1 <= n2) {
      //        ranking -= (newP1 - p1 + 1);
      //        p1 = newP1 + 1;
      //      } else {
      //        ranking -= (newP2 - p2 + 1);
      //        p2 = newP2 + 1;
      //      }
    }
    if (p1 == l1) return nums2[p2 - ranking + 1];
    if (p2 == l2) return nums1[p1 - ranking + 1];
    return Math.max(nums1[p1], nums2[p2]);
    //    if (p1 == l1) return nums2[p2 + ranking - 1];
    //    if (p2 == l2) return nums1[p1 + ranking - 1];
    //    return Math.min(nums1[p1], nums2[p2]);
  }

  // 尾递归，特判 k=1 & 其一为空
  private int getkSmallElement(
      int[] nums1, int lo1, int hi1, int[] nums2, int lo2, int hi2, int k) {
    if (k == 1) return Math.min(nums1[lo1], nums2[lo2]);
    int l1 = hi1 - lo1 + 1, l2 = hi2 - lo2 + 1;
    // 让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1
    if (l1 > l2) {
      return getkSmallElement(nums2, lo2, hi2, nums1, lo1, hi1, k);
    } else if (l1 == 0) {
      return nums2[lo2 + k - 1];
    }
    int p1 = lo1 + Math.min(l1, k / 2) - 1, p2 = lo2 + Math.min(l2, k / 2) - 1;
    return (nums1[p1] > nums2[p2])
        ? getkSmallElement(nums1, lo1, hi1, nums2, p2 + 1, hi2, k - (p2 - lo2 + 1))
        : getkSmallElement(nums1, p1 + 1, hi1, nums2, lo2, hi2, k - (p1 - lo1 + 1));
  }

  /**
   * 在排序数组中查找元素的第一个和最后一个位置
   *
   * <p>参考
   * https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/si-lu-hen-jian-dan-xi-jie-fei-mo-gui-de-er-fen-cha/
   *
   * @param nums the nums
   * @param target the target
   * @return int [ ]
   */
  public int[] searchRange(int[] nums, int target) {
    if (target < nums[0] || nums[nums.length - 1] < target) {
      return new int[] {-1, -1};
    }
    int smallmost = smallmostBound(nums, target);
    if (smallmost == -1) return new int[] {-1, -1};
    int bigmost = bigmostBound(nums, target, smallmost);
    return new int[] {smallmost, bigmost};
  }

  /**
   * 对有序数组找到重复数超过 k 的序列
   *
   * <p>二分找到某数出现的首个和最后的索引，再分隔两个数组分别递归求解。
   *
   * @param nums
   * @param k
   * @return
   */
  public List<Integer> findDuplicatesK(int[] nums, int k) {
    List<Integer> res = new ArrayList<>();
    int[] pos = searchRange(nums, nums[nums.length / 2]);
    List<Integer> p1 = findDuplicatesK(Arrays.copyOfRange(nums, 0, pos[0]), k),
        p2 = findDuplicatesK(Arrays.copyOfRange(nums, pos[1], nums.length), k);
    return res;
  }

  /**
   * 搜索旋转排序数组，目标分别与中点，左右边界对比，有序的一边的边界值可能等于目标值
   *
   * <p>将数组一分为二，其中一定有一个是有序的，另一个可能是有序，此时有序部分用二分法查找，无序部分再一分为二，其中一个一定有序，另一个可能有序
   *
   * <p>33-查找旋转数组不重复；81-查找旋转数组可重复复；153-旋转数组最小值不重复；154旋转数字最小值重复
   *
   * <p>扩展1，旋转 k 次，无论旋转几次，最多只有俩段递增序列
   *
   * @param nums the nums
   * @param target the target
   * @return int int
   */
  public int search(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo + 1) / 2; // 向上取整
      if (nums[mid] < nums[hi]) {
        // target 落在 [mid, hi]
        if (nums[mid] <= target && target <= nums[hi]) lo = mid;
        else hi = mid - 1;
      } else {
        // target 落在 [lo, mid-1]
        if (nums[lo] <= target && target <= nums[mid - 1]) hi = mid - 1;
        else lo = mid;
      }
    }
    return nums[lo] == target ? lo : -1;
  }

  /**
   * 寻找旋转排序数组中的最小值，无重复，比较边界
   *
   * <p>扩展1，找最大，可复用本题，参考「山脉数组的顶峰索引」
   *
   * <p>扩展2，寻找旋转排序数组中的最小值II，有重复，参下 annotate
   *
   * <p>扩展3，降序且旋转的数组，求最小值
   *
   * @param nums the nums
   * @return int int
   */
  public int findMin(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (nums[mid] < nums[hi]) hi = mid;
      // 有重复，则判断
      // else if (nums[mid] == nums[hi]) hi -= 1;
      else lo = mid + 1;
    }
    return nums[lo];
  }

  /**
   * 寻找峰值，返回任意一个峰的索引
   *
   * <p>扩展1，需要返回的峰索引满足左右均单调递增，山脉数组的顶峰索引，参下 annotate
   *
   * @param nums the nums
   * @return int int
   */
  public int findPeakElement(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    // int lo = 1, hi = nums.length - 2; // peakIndexInMountainArray
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (nums[mid] < nums[mid + 1]) lo = mid + 1; // 缩减区间为 [mid+1, hi]
      else hi = mid;
    }
    return lo; // 碰撞时结束
  }

  /**
   * 山脉数组中查找目标值，三段二分
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/find-in-mountain-array/solution/shi-yong-chao-hao-yong-de-er-fen-fa-mo-ban-python-/
   *
   * @param target
   * @param mountainArr
   * @return
   */
  public int findInMountainArray(int target, MountainArray mountainArr) {
    int lo = 0, hi = mountainArr.length() - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (mountainArr.get(mid) < mountainArr.get(mid + 1)) lo = mid + 1;
      else hi = mid;
    }
    // 模板同上「二分查找」最终需要检查左边界合法性
    int peak = lo, idx = search(mountainArr, target, 0, peak, true);
    return idx != -1 ? idx : search(mountainArr, target, peak + 1, mountainArr.length() - 1, false);
  }

  private int search(MountainArray mountainArr, int target, int lo, int hi, boolean flag) {
    target *= flag ? 1 : -1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2, cur = mountainArr.get(mid) * (flag ? 1 : -1);
      if (cur < target) lo = mid + 1;
      else if (cur == target) return mid;
      else hi = mid;
    }
    int mid = lo + (hi - lo) / 2, cur = mountainArr.get(mid) * (flag ? 1 : -1);
    return cur == target ? lo : -1;
  }
}

class DichotomyElse extends DefaultArray {
  /**
   * 搜索二维矩阵，确保每行的首个大于前一行的最后一个
   *
   * <p>行内有序，行间不重叠。否则，参考「有序矩阵中第k小的元素」
   *
   * <p>模拟 BST 以右上角作根开始遍历则 I & II 通用
   *
   * <p>扩展1，II 不确保「每行的首个大于前一行的最后一个」，因此无法两次二分，只能遍历行/列，再对列/行进行二分
   *
   * @param matrix the matrix
   * @param target the target
   * @return boolean boolean
   */
  public boolean searchMatrix(int[][] matrix, int target) {
    // 定位到所在行，找到最后一个满足 matrix[x]][0] <= t 的行
    int lo = 0, hi = matrix.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo + 1) / 2;
      if (matrix[mid][0] <= target) lo = mid;
      else hi = mid - 1;
    }
    int row = hi, col;
    if (matrix[row][0] == target) return true;
    if (matrix[row][0] > target) return false;
    // 从所在行中定位到列，找到最后一个满足 matrix[row][x] <= t 的列
    col = bigmostBound(matrix[row], target, 0);
    return col == -1 ? false : matrix[row][col] == target;
    // II
    //    for (int row = 0; row < matrix.length; row++) {
    //    int col = bigmostBound(matrix[row], target, 0);
    //    if (col != -1 && matrix[row][col] == target) return true;
    //    }
    //    return false;
  }

  /**
   * 有序矩阵中第k小的元素，题设元素唯一，因此对数值进行二分，即值域
   *
   * <p>只保证行内与单列有序，意味着行间区间可能重叠。
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/solution/er-fen-chao-ji-jian-dan-by-jacksu1024/
   *
   * @param matrix
   * @param k
   * @return
   */
  public int kthSmallest(int[][] matrix, int k) {
    // 左上角与右下角即数值的上下界
    int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1];
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      // 找二维矩阵中 <=mid 的元素总个数，判断目标分别在 [lo,mid] or [mid+1,hi]
      if (countLte(matrix, mid) < k) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // 从左下角开始遍历，找每列最后一个 <=target 的数即知道每一列有多少个数 <=target
  private int countLte(int[][] matrix, int target) {
    int count = 0, x = 0, y = matrix.length - 1;
    while (-1 < y && x < matrix[0].length) {
      if (matrix[y][x] <= target) {
        // 第 j 列有 i+1 个元素 <= mid
        count += y + 1;
        x += 1;
      } else {
        // 第 j 列目前的数大于 mid，需要继续在当前列往上找
        y -= 1;
      }
    }
    return count;
  }

  /**
   * 有序数组中的单一元素，无序也适用
   *
   * <p>假设所有数字都成对，那么所有数字的下标必定同时偶数和奇数，因此比对 nums[mid]
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/single-element-in-a-sorted-array/solution/er-fen-fa-wu-xu-shu-zu-ye-gua-yong-by-li-s6f2/
   *
   * @param nums
   * @return
   */
  public int singleNonDuplicate(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      // 移除中心元素后，其右侧的数量为奇数，如 1144 5 5688
      if (mid % 2 == 0 && nums[mid] == nums[mid + 1]) lo = mid + 2;
      else if (mid % 2 == 1 && nums[mid] == nums[mid - 1]) lo = mid + 1; // 同上，但如 11445 5 66899
      else hi = mid; // 取中心值时会向下取整，如 11455 6 68899 & 1145 5 6688
    }
    return nums[lo];
  }

  /**
   * 和为s的连续正数序列
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/solution/shi-yao-shi-hua-dong-chuang-kou-yi-ji-ru-he-yong-h/
   *
   * @param target
   * @return
   */
  public int[][] findContinuousSequence(int target) {
    List<int[]> seq = new ArrayList<int[]>();
    int lo = 1, hi = 2;
    while (lo < hi) {
      // 区间求和公式
      int sum = (lo + hi) * (hi - lo + 1) / 2;
      if (sum < target) {
        hi += 1;
      } else if (sum == target) {
        int[] ans = new int[hi - lo + 1];
        for (int i = lo; i <= hi; i++) {
          ans[i - lo] = i;
        }
        seq.add(ans);
        lo += 1;
      } else {
        lo += 1;
      }
    }
    return seq.toArray(new int[seq.size()][]);
  }

  /**
   * 分割数组的最大值
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/split-array-largest-sum/solution/er-fen-cha-zhao-by-liweiwei1419-4/
   *
   * @param nums
   * @param m
   * @return
   */
  public int splitArray(int[] nums, int m) {
    int max = 0, sum = 0;
    // 计算子数组各自和的最大值的上下界
    for (int n : nums) {
      max = Math.max(max, n);
      sum += n;
    }
    // 二分确定一个恰当的子数组各自的和的最大值，使得它对应的「子数组的分割数」恰好等于 m
    int lo = max, hi = sum;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2, splits = split(nums, mid);
      // 如果分割数太多，说明「子数组各自的和的最大值」太小，此时需要调大该值
      if (splits > m) lo = mid + 1; // 下一轮搜索的区间是 [mid + 1, right]
      else hi = mid;
    }
    return lo;
  }

  // 满足不超过「子数组各自的和的最大值」的分割数
  private int split(int[] nums, int maxIntervalSum) {
    // 至少是一个分割 & 当前区间的和
    int splits = 1, curIntervalSum = 0;
    for (int n : nums) {
      // 尝试加上当前遍历的这个数，如果加上去超过了「子数组各自的和的最大值」，就不加这个数，另起炉灶
      if (curIntervalSum + n > maxIntervalSum) {
        curIntervalSum = 0;
        splits += 1;
      }
      curIntervalSum += n;
    }
    return splits;
  }
}

/**
 * 区间和相关，主要运用前缀和，参考
 * https://leetcode.cn/problems/corporate-flight-bookings/solution/gong-shui-san-xie-yi-ti-shuang-jie-chai-fm1ef/
 */
class SSum extends DefaultArray {
  /**
   * 三数之和
   *
   * @param nums the nums
   * @return list list
   */
  public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> sums = new ArrayList();
    int target = 0;
    Arrays.sort(nums);
    for (int i = 0; i < nums.length; i++) {
      int pivot = nums[i];
      if (pivot > target) break;
      if (i > 0 && pivot == nums[i - 1]) continue; // first deduplication
      int lo = i + 1, hi = nums.length - 1; // two sum
      while (lo < hi) {
        int sum = pivot + nums[lo] + nums[hi];
        if (sum < target) {
          lo += 1;
        } else if (sum > target) {
          hi -= 1;
        } else if (sum == target) {
          sums.add(Arrays.asList(pivot, nums[lo], nums[hi]));
          while (lo < hi && nums[lo] == nums[lo + 1]) { // second deduplication
            lo += 1;
          }
          while (lo < hi && nums[hi] == nums[hi - 1]) {
            hi -= 1;
          }
          lo += 1;
          hi -= 1;
        }
      }
    }
    return sums;
  }

  /**
   * 最接近的三数之和，题设解唯一
   *
   * @param nums the nums
   * @param target the target
   * @return int
   */
  public int threeSumClosest(int[] nums, int target) {
    Arrays.sort(nums);
    int closestSum = nums[0] + nums[1] + nums[2]; // 题设至少三位
    for (int i = 0; i < nums.length; i++) {
      int pivot = nums[i];
      if (pivot > target) break;
      int lo = i + 1, hi = nums.length - 1;
      while (lo < hi) {
        int sum = pivot + nums[lo] + nums[hi];
        if (Math.abs(target - sum) < Math.abs(target - closestSum)) closestSum = sum;
        if (sum < target) lo += 1;
        else if (sum == target) return closestSum;
        else hi -= 1;
      }
    }
    return closestSum;
  }

  /**
   * 将数组分成和相等的三部分，无序数组，且有正负
   *
   * <p>求和 & 特判 & 二分判断左右区间总和是否为 sum/3
   *
   * <p>参考
   * https://leetcode-cn.com/problems/partition-array-into-three-parts-with-equal-sum/solution/java-shi-yong-shuang-zhi-zhen-by-sugar-31/
   *
   * @param arr
   * @return
   */
  public boolean canThreePartsEqualSum(int[] nums) {
    int sum = 0;
    for (int i : nums) {
      sum += i;
    }
    // 特判总和非 3 的倍数
    if (sum % 3 != 0) return false;
    int lo = 0, hi = nums.length - 1;
    int loSum = nums[lo], hiSum = nums[hi];
    // 通过 left+1<right 防止只能将数组分成两个部分
    // 如 [1,-1,1,-1]，使用 left<right 作为判断条件就会出错
    while (lo + 1 < hi) {
      // 左右两边都等于 sum/3 ，中间也一定等于
      if (loSum == sum / 3 && hiSum == sum / 3) return true;
      if (loSum != sum / 3) {
        lo += 1;
        loSum += nums[lo];
      }
      if (hiSum != sum / 3) {
        hi -= 1;
        hiSum += nums[hi];
      }
    }
    return false;
  }

  /**
   * 有效三角形的个数，类似三数之和
   *
   * <p>参考
   * https://leetcode.cn/problems/valid-triangle-number/solution/ming-que-tiao-jian-jin-xing-qiu-jie-by-jerring/
   *
   * @param nums
   * @return
   */
  public int triangleNumber(int[] nums) {
    Arrays.sort(nums);
    int count = 0;
    for (int i = nums.length - 1; i >= 2; i--) {
      int pivot = nums[i], lo = 0, hi = i - 1;
      while (lo < hi) {
        if (nums[lo] + nums[hi] > pivot) {
          count += hi - lo;
          hi -= 1;
        } else {
          lo += 1;
        }
      }
    }
    return count;
  }

  /**
   * 四数之和
   *
   * <p>TODO
   *
   * @param nums
   * @param target
   * @return
   */
}

/**
 * 以下均为前缀和，适合区间和满足 target
 *
 * <p>1.严格相等搭配哈希，否则滑窗
 *
 * <p>2.允许一边遍历求，则只存总和，否则需要区间和
 *
 * <p>参考
 * https://leetcode.cn/problems/subarray-sum-equals-k/solution/de-liao-yi-wen-jiang-qian-zhui-he-an-pai-yhyf/
 */
class PreSum {
  /**
   * 最大子数组和/最大子序和/最大子串和/连续子数组的最大和，基于贪心，通过前缀和
   *
   * <p>dp[i] 表示以 nums[i] 结尾的最大子序和，状态压缩为 curSum
   *
   * <p>sum>0 说明 sum 对结果有增益效果，则后者保留并加上当前遍历数字，否则舍弃，sum 直接更新为当前遍历数字
   *
   * <p>参考
   * https://leetcode-cn.com/problems/maximum-subarray/solution/53zui-da-zi-xu-he-tan-xin-de-qian-zhui-h-aov9/
   *
   * <p>线段树，参考 education in codeforces
   *
   * <p>扩展1，返回左右边界，或该子数组，则添加始末指针，参下
   *
   * @param nums the nums
   * @return int int
   */
  public int maxSubArray(int[] nums) {
    int preSum = 0, maxSum = Integer.MIN_VALUE;
    // int start = 0, end = 0;
    for (int n : nums) {
      if (preSum + n > n) {
        preSum += n;
      } else {
        preSum = n;
        //        start = i;
      }
      if (preSum > maxSum) {
        maxSum = preSum;
        //        end = i;
      }
    }
    return maxSum;
  }

  /**
   * 和为k的子数组，返回其数量，严格相等
   *
   * <p>设 [i:j] 子数组和为 k，则有 pre[j−1] == pre[i]-k，因此计数 pre[j−1] 即可
   *
   * <p>参考
   * https://leetcode-cn.com/problems/subarray-sum-equals-k/solution/de-liao-yi-wen-jiang-qian-zhui-he-an-pai-yhyf/
   *
   * <p>扩展1，求本题结果集中最大的长度，参考「连续数组」
   *
   * @param nums the nums
   * @param k the k
   * @return int int
   */
  public int subarraySum(int[] nums, int k) {
    int count = 0, preSum = 0;
    Map<Integer, Integer> sum2Count = new HashMap<>();
    // 需要预存 0 否则会漏掉前几位就满足的情况
    // 如 [1,1,0]，k=2 会返回 0 漏掉 1+1=2 与 1+1+0=2
    // 输入 [3,1,1,0] k=2 时则不会漏掉，因为 presum[3]-presum[0] 表示前面 3 位的和
    sum2Count.put(0, 1);
    for (int n : nums) {
      preSum += n;
      if (sum2Count.containsKey(preSum - k)) count += sum2Count.get(preSum - k);
      sum2Count.put(preSum, sum2Count.getOrDefault(preSum, 0) + 1);
    }
    return count;
  }

  /**
   * 连续数组，严格相等，最长，找到数量相同的 0 和 1 的最长子数组，题设非 0 即 1
   *
   * <p>将 0 作为 −1，则转换为求区间和满足 0 的最长子数组，同时记录「某个前缀和出现的首个右边界」
   *
   * <p>参考
   * https://leetcode-cn.com/problems/contiguous-array/solution/qian-zhui-he-chai-fen-ha-xi-biao-java-by-liweiwei1/
   *
   * @param nums the nums
   * @return int
   */
  public int findMaxLength(int[] nums) {
    int preSum = 0, maxLen = 0;
    Map<Integer, Integer> sum2FirstHi = new HashMap<>();
    sum2FirstHi.put(0, -1); // 可能存在前缀和刚好满足条件的情况
    for (int i = 0; i < nums.length; i++) {
      // 将 0 作为 -1
      preSum += nums[i] == 0 ? -1 : 1;
      // 画图可知 i - map[preSum] 即和为 0 的数组的长度
      if (sum2FirstHi.containsKey(preSum)) maxLen = Math.max(maxLen, i - sum2FirstHi.get(preSum));
      // 仍未遍历到和为 0 的子数组，更新即可
      else sum2FirstHi.put(preSum, i);
    }
    return maxLen;
  }

  /**
   * 和可被k整除的子数组，严格相等
   *
   * <p>参考
   * https://leetcode-cn.com/problems/subarray-sum-equals-k/solution/de-liao-yi-wen-jiang-qian-zhui-he-an-pai-yhyf/
   *
   * @param A
   * @param K
   * @return
   */
  public int subarraysDivByK(int[] nums, int k) {
    int preSum = 0, count = 0;
    HashMap<Integer, Integer> remainder2Count = new HashMap<>();
    remainder2Count.put(0, 1);
    for (int num : nums) {
      preSum += num;
      // 当前 presum 与 K 的关系，余数是几，当被除数为负数时取模结果为负数，需要纠正
      int remainder = (preSum % k + k) % k, curCount = remainder2Count.getOrDefault(remainder, 0);
      // 余数的次数
      count += curCount;
      remainder2Count.put(remainder, curCount + 1);
    }
    return count;
  }

  // 以下均需要区间和，因此前缀和为数组。

  /**
   * 连续的子数组和，返回是否存在子数组满足总和为 k 的倍数，且至少有两个元素，严格相等
   *
   * <p>只需要枚举右端点 j，然后在枚举右端点 j 时检查之前是否出现过左端点 i，使得 sum[j] & sum[i - 1] 对 k 取余相同
   *
   * <p>扩展1，方案数参考
   * https://leetcode-cn.com/problems/continuous-subarray-sum/solution/gong-shui-san-xie-tuo-zhan-wei-qiu-fang-1juse/
   *
   * @param nums the nums
   * @param target the target
   * @return boolean
   */
  public boolean checkSubarraySum(int[] nums, int target) {
    int len = nums.length;
    int[] preSum = new int[len + 1]; // 哑元素，因此下方始于 2
    for (int i = 1; i <= len; i++) {
      preSum[i] = preSum[i - 1] + nums[i - 1];
    }
    Map<Integer, Integer> visted = new HashMap<>();
    for (int i = 2; i <= len; i++) {
      visted.put(preSum[i - 2] % target, 0); // can be replaced with HashSet
      if (visted.containsKey(preSum[i] % target)) return true;
    }
    return false;
    // 方案数
    //    int count = 0;
    //    remainder2Count.put(0, 1);
    //    for (int i = 1; i <= nums.length; i++) {
    //      int mod = preSum[i] % target;
    //      remainder2Count.put(mod, remainder2Count.getOrDefault(mod, 0) + 1);
    //      if (remainder2Count.containsKey(mod)) count += remainder2Count.get(mod);
    //    }
  }

  /**
   * 和至少为k的最短子数组，返回长度，和至少，前缀和数组 & 单调队列
   *
   * <p>此处入队索引而「滑动窗口的最大值」则是值
   *
   * <p>TODO 需要找到索引 x & y 使得 prefix[y]-prefix[x]>=k 且 y-x 最小
   *
   * <p>参考
   * https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/solution/he-zhi-shao-wei-k-de-zui-duan-zi-shu-zu-by-leetcod/
   *
   * <p>扩展1，最长，参考「表现良好的最长时间段」，将 <8 视作 -1 否则视作 1，找和为正数的最长子数组
   *
   * @param nums the nums
   * @param k the k
   * @return int int
   */
  public int shortestSubarray(int[] nums, int k) {
    int len = nums.length, minLen = len + 1;
    long[] preSum = new long[len + 1];
    for (int i = 0; i < len; i++) {
      preSum[i + 1] = preSum[i] + nums[i];
    }
    Deque<Integer> mq = new LinkedList<>();
    for (int i = 0; i < preSum.length; i++) {
      long sum = preSum[i];
      while (!mq.isEmpty() && preSum[mq.getLast()] >= sum) {
        mq.removeLast();
      }
      while (!mq.isEmpty() && preSum[mq.getFirst()] + k <= sum) {
        minLen = Math.min(minLen, i - mq.removeFirst());
      }
      mq.addLast(i);
    }
    return minLen == len + 1 ? -1 : minLen;
  }

  /**
   * 统计「优美子数组」，区间包含 k 个奇数
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/subarray-sum-equals-k/solution/de-liao-yi-wen-jiang-qian-zhui-he-an-pai-yhyf/
   *
   * @param nums
   * @param k
   * @return
   */
  public int numberOfSubarrays(int[] nums, int k) {
    // preSum
    int len = nums.length, countOdd = 0, count = 0;
    int[] map = new int[len + 1];
    map[0] = 1;
    for (int i = 0; i < len; ++i) {
      // 如果是奇数则加 1 偶数加 0
      countOdd += nums[i] & 1;
      if (countOdd - k >= 0) count += map[countOdd - k];
      map[countOdd] += 1;
    }
    return count;
  }

  /**
   * 最大子矩阵，元素和最大，类似「最大子数组和」
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/max-submatrix-lcci/solution/zhe-yao-cong-zui-da-zi-xu-he-shuo-qi-you-jian-dao-/
   *
   * @param matrix the matrix
   * @return int [ ]
   */
  public int[] getMaxMatrix(int[][] matrix) {
    int row = matrix.length, col = matrix[0].length;
    // 保存最大子矩阵的左上角和右下角的行列坐标 & 记录当前 i~j 行组成大矩阵的每一列的和，将二维转化为一维
    int[] coordinates = new int[4], preSum = new int[col];
    int sum = 0, maxSum = Integer.MIN_VALUE; // 相当于dp[i],dp_i 与最大值
    int bestr1 = 0, bestc1 = 0; // 暂时记录左上角，相当于 start
    for (int i = 0; i < row; i++) { // 以i为上边，从上而下扫描
      for (int t = 0; t < col; t++) {
        preSum[t] = 0; // 每次更换子矩形上边，就要清空b，重新计算每列的和
      }
      for (int j = i; j < row; j++) {
        // 子矩阵的下边，从i到N-1，不断增加子矩阵的高，相当于求一次最大子序列和
        sum = 0; // 从头开始求dp
        for (int k = 0; k < col; k++) {
          preSum[k] += matrix[j][k];
          // 我们只是不断增加其高，也就是下移矩阵下边，所有这个矩阵每列的和只需要加上新加的哪一行的元素
          // 因为我们求dp[i]的时候只需要dp[i-1]和nums[i],所有在我们不断更新b数组时就可以求出当前位置的dp_i
          if (sum > 0) {
            sum += preSum[k];
          } else {
            sum = preSum[k];
            bestr1 = i; // 自立门户，暂时保存其左上角
            bestc1 = k;
          }
          if (sum > maxSum) {
            maxSum = sum;
            coordinates = new int[] {bestr1, bestc1, j, k};
          }
        }
      }
    }
    return coordinates;
  }
}

/** 重复，原地哈希 */
class DDuplicate extends DefaultArray {
  /**
   * 寻找重复数，无序找首个，快慢指针，类似「环形链表II」，可参考「第一个只出现一次的字符」
   *
   * <p>参考
   * https://leetcode-cn.com/problems/find-the-duplicate-number/solution/kuai-man-zhi-zhen-de-jie-shi-cong-damien_undoxie-d/
   *
   * <p>TODO 扩展1，重复数字有多个，找出所有，要求复杂度 n & 1
   *
   * <p>nums[i] 每出现过一次对 nums[idx]+=n，其中 idx=nums[i]-1，加完之后，当nums[idx]>2*n 时就能表示 nums[i]，即 idx+1
   * 出现过两次
   *
   * @param nums the nums
   * @return int int
   */
  public int findDuplicate(int[] nums) {
    int lo = nums[0], hi = nums[nums[0]]; // 题设区间为 [1,n]
    while (lo != hi) {
      lo = nums[lo];
      hi = nums[nums[hi]];
    }
    int finder = 0;
    while (lo != finder) {
      lo = nums[lo];
      finder = nums[finder];
    }
    return lo;
  }

  /**
   * 数组中重复的数字，返回任意一个重复的数字，区间 [0,nums.length-1]
   *
   * <p>原地哈希，i 需要命中 nums[i]，即将整个数组排序，理应是 nums[i]=i
   *
   * @param nums
   * @return
   */
  public int findRepeatNumber(int[] nums) {
    for (int i = 0; i < nums.length; i++) {
      while (nums[i] != i) {
        if (nums[i] == nums[nums[i]]) return nums[i];
        swap(nums, i, nums[i]);
      }
    }
    return -1;
  }

  /**
   * 数组中重复的数据，返回所有重复数，每个数字至多出现两次，区间 [1,n]
   *
   * <p>原地哈希，重复会命中同一索引，nums[nums[i]-1]*=-1，类似缺失的第一个整数
   *
   * @param nums the nums
   * @return list list
   */
  public List<Integer> findDuplicates(int[] nums) {
    List<Integer> duplicates = new ArrayList<>();
    for (int n : nums) {
      int idx = (n < 0 ? -n : n) - 1;
      if (nums[idx] < 0) duplicates.add(idx + 1); // visite num marked
      else nums[idx] *= -1; // mark num
    }
    return duplicates;
  }

  /**
   * 缺失的第一个正数
   *
   * <p>原地哈希，缺失会命中错误索引，nums[nums[i]-1]!=nums[i]，类似数组中重复的数据
   *
   * @param nums the nums
   * @return int int
   */
  public int firstMissingPositive(int[] nums) {
    int len = nums.length;
    for (int i = 0; i < len; i++) {
      // 不断判断 i 位置上被放入正确的数 nums[i]-1
      while (nums[i] > 0 && nums[i] <= len && nums[i] != nums[nums[i] - 1]) {
        swap(nums, nums[i] - 1, i);
      }
    }
    for (int i = 0; i < len; i++) {
      if (nums[i] != i + 1) return i + 1;
    }
    return len + 1; // 没有缺失
  }
}

/** 移除相关，类似滑窗 */
class Delete extends DefaultArray {
  /**
   * 调整数组顺序使奇数位于偶数前面，参考移动零，即遇到目标则跳过
   *
   * <p>扩展1，链表参考「奇偶链表」
   *
   * @param nums
   * @return
   */
  public int[] exchange(int[] nums) {
    int write = 0;
    for (int read = 0; read < nums.length; read++) {
      if ((nums[read] & 1) == 0) continue;
      swap(nums, write, read);
      write += 1;
    }
    return nums;
  }

  /**
   * 移动零，遇到目标则跳过
   *
   * <p>扩展1，移除字符串中指定字符，同模板，参下
   *
   * <p>扩展2，a & b 移至末尾，且所有元素保持相对顺序，参下 annotate
   *
   * @param nums the nums
   */
  public void moveZeroes(int[] nums) {
    final int target = 0, k = 0;
    int write = 0;
    for (int read = 0; read < nums.length; read++) {
      //      if (nums[hi] != 1 && nums[hi] != 6 && nums[hi] != 3)
      if (write >= k && target == nums[read]) continue;
      swap(nums, write, read);
      write += 1;
    }
  }

  /**
   * 移除字符串中指定字符
   *
   * @param str
   * @param target
   * @return
   */
  public String moveChars(String str, char target) {
    char[] chs = str.toCharArray();
    int write = 0;
    for (int read = 0; read < chs.length; read++) {
      if (target == chs[read]) continue;
      swap(chs, write, read);
      write += 1;
    }
    return String.valueOf(Arrays.copyOfRange(chs, 0, write));
  }

  /**
   * 删除排序数组中的重复项，保留 k 位，I & II 通用
   *
   * <p>原地，解法等同移动零，需要移除的目标位 nums[last - k]
   *
   * <p>扩展1，参考删除字符串中的所有相邻重复项
   *
   * @param nums the nums
   * @return the int
   */
  public int removeDuplicates(int[] nums) {
    final int k = 0;
    // final int target = nums[last - k];
    int write = 0;
    for (int read = 0; read < nums.length; read++) {
      int num = nums[read];
      if (write >= k && nums[write - k] == num) continue;
      nums[write] = num;
      write += 1;
    }
    return write;
  }

  /**
   * 删除字符串中的所有相邻重复项，毫无保留，原地模拟栈
   *
   * <p>类似「有效的括号」即括号匹配，通过 top 指针模拟栈顶，即原地栈，且修改源数组
   *
   * <p>匹配指当前字符与栈顶不同，即入栈，否则出栈，且 skip 当前 char
   *
   * <p>最终栈内即为最终结果
   *
   * <p>参考
   * https://leetcode.cn/problems/remove-all-adjacent-duplicates-in-string/solution/tu-jie-guan-fang-tui-jian-ti-jie-shan-ch-x8iz/
   *
   * @param s the s
   * @return string string
   */
  public String removeDuplicates(String s) {
    char[] chs = s.toCharArray();
    int top = -1;
    for (char ch : chs) {
      if (top > -1 && chs[top] == ch) {
        top -= 1;
      } else {
        top += 1;
        chs[top] = ch;
      }
    }
    return String.valueOf(chs, 0, top + 1);
  }
}

/** 遍历相关 */
class Traversal extends DefaultArray {
  /**
   * 轮转数组 / 旋转数组，反转三次，the whole & [0,k-1] & [k,end]
   *
   * @param nums the nums
   * @param k the k
   */
  public void rotate(int[] nums, int k) {
    k %= nums.length;
    reverse(nums, 0, nums.length - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.length - 1);
  }

  /**
   * 旋转图像 / 旋转矩阵
   *
   * <p>沿东南斜对角线 & 垂直中线翻转
   *
   * <p>扩展1，翻转 180 度，则分别沿水平与垂直翻转，而 270 度则改为沿西南斜对角线
   *
   * @param matrix
   */
  public void rotate(int[][] matrix) {
    int len = matrix.length;
    for (int y = 0; y < len; y++) {
      for (int x = 0; x < y; x++) {
        // swap (y,x) and (x,y)
        int tmp = matrix[y][x];
        matrix[y][x] = matrix[x][y];
        matrix[x][y] = tmp;
      }
    }
    for (int i = 0; i < len; i++) {
      reverse(matrix[i], 0, len - 1);
    }
  }

  /**
   * 螺旋矩阵，遍历
   *
   * @param matrix
   * @return
   */
  public List<Integer> spiralOrder(int[][] matrix) {
    int row = matrix.length, col = matrix[0].length;
    List<Integer> res = new ArrayList<>(row * col);
    int up = 0, down = row - 1, left = 0, right = col - 1;
    while (true) { // 任何一组越界即结束遍历
      for (int i = left; i <= right; i++) {
        res.add(matrix[up][i]);
      }
      up += 1;
      if (up > down) break;
      for (int i = up; i <= down; i++) {
        res.add(matrix[i][right]);
      }
      right -= 1;
      if (right < left) break;
      for (int i = right; i >= left; i--) {
        res.add(matrix[down][i]);
      }
      down -= 1;
      if (down < up) break;
      for (int i = down; i >= up; i--) {
        res.add(matrix[i][left]);
      }
      left += 1;
      if (left > right) break;
    }
    return res;
  }

  /**
   * 螺旋矩阵II，生成
   *
   * <p>left,right & up,down & right,left & down & up
   *
   * @param n
   * @return
   */
  public int[][] generateMatrix(int n) {
    int[][] res = new int[n][n];
    int num = 1;
    int left = 0, right = n - 1, up = 0, down = n - 1;
    while (num <= n * n) {
      for (int i = left; i <= right; i++) {
        res[up][i] = num;
        num += 1;
      }
      up += 1;
      for (int i = up; i <= down; i++) {
        res[i][right] = num;
        num += 1;
      }
      right -= 1;
      for (int i = right; i >= left; i--) {
        res[down][i] = num;
        num += 1;
      }
      down -= 1;
      for (int i = down; i >= up; i--) {
        res[i][left] = num;
        num += 1;
      }
      left += 1;
    }
    return res;
  }

  /**
   * 字符的最短距离，返回 answer[i] 是 s[i] 与所有 s.chatAt(c) 的最小值
   *
   * <p>依次正序和逆序遍历，分别找出距离向左或者向右下一个字符 C 的距离，答案就是这两个值的较小值
   *
   * <p>参考
   * https://leetcode-cn.com/problems/shortest-distance-to-a-character/solution/zi-fu-de-zui-duan-ju-chi-by-leetcode/
   *
   * @param s
   * @param c
   * @return
   */
  public int[] shortestToChar(String s, char c) {
    int len = s.length(), pre = Integer.MIN_VALUE / 2;
    int[] minDistances = new int[len];
    for (int i = 0; i < len; i++) {
      if (s.charAt(i) == c) pre = i;
      minDistances[i] = i - pre;
    }
    pre = Integer.MAX_VALUE / 2;
    for (int i = len - 1; i >= 0; i--) {
      if (s.charAt(i) == c) pre = i;
      minDistances[i] = Math.min(minDistances[i], pre - i);
    }
    return minDistances;
  }

  /**
   * 对角线遍历
   *
   * <p>扩展1，反对角线，则将下方 bXFlag 初始为 false
   *
   * @param matrix
   * @return
   */
  public int[] findDiagonalOrder(int[][] matrix) {
    if (matrix == null || matrix.length == 0) {
      return new int[0];
    }
    int m = matrix.length, n = matrix[0].length;
    int[] res = new int[m * n];
    // 当前共遍历 k 个元素
    int k = 0;
    // 当前是否东北向遍历，反之为西南向
    boolean bXFlag = true;
    // 共有 m+n-1 条对角线
    for (int i = 0; i < m + n - 1; i++) {
      int pm = bXFlag ? m : n, pn = bXFlag ? n : m;
      int x = (i < pm) ? i : pm - 1, y = i - x;
      while (x >= 0 && y < pn) {
        res[k] = bXFlag ? matrix[x][y] : matrix[y][x];
        k += 1;
        x -= 1;
        y += 1;
      }
      bXFlag = !bXFlag;
    }
    return res;
  }
}

/** 字典序相关 */
class DicOrder extends DefaultArray {
  /**
   * 下一个排列，求按照字典序，该排列下一个大的
   *
   * <p>对比下方「最大交换」，后者是找到交换结果的最大
   *
   * <p>扩展1，上一个排列，从 n-2 开始找到首个峰 & 峰右边调为降序 & 从 n-1 开始找到首个比峰小的数，交换
   *
   * @param nums the nums
   */
  public void nextPermutation(int[] nums) {
    int peak = nums.length - 1; // nums.length - 2
    while (peak > 0) {
      // 1.find the first peak and sort from its idx to end
      if (nums[peak] > nums[peak - 1]) {
        Arrays.sort(nums, peak, nums.length);
        break;
      }
      peak -= 1;
    }
    // 2.find the second peak larger than IDX-1 and swap them
    for (int j = peak; j < nums.length; j++) {
      if (nums[j] <= nums[peak - 1]) continue;
      swap(nums, peak - 1, j);
      return;
    }
    // 3.Otherwise, monotonic without peaks
    Arrays.sort(nums);
  }

  /**
   * 最大交换，只交换一次任意两位的数，使其结果的数值是所有方案中最大的
   *
   * <p>贪心，将较高位的 n 与后面 m 交换，需满足 m>n 且 m 尽可能靠后
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/maximum-swap/solution/2021316-zui-da-jiao-huan-quan-chang-zui-ery0x/
   *
   * @param num
   * @return
   */
  public int maximumSwap(int num) {
    char[] chs = Integer.toString(num).toCharArray();
    int[] lastIdxes = new int[10]; // 每种数字最后的索引
    for (int i = 0; i < chs.length; i++) {
      lastIdxes[chs[i] - '0'] = i;
    }
    // 查找首个值更大、位更高的数字
    for (int i = 0; i < chs.length; i++) { // 自高位顺序遍历
      for (int d = 9; d > chs[i] - '0'; d--) { // 值
        if (lastIdxes[d] <= i) continue; // 位
        swap(chs, i, lastIdxes[d]);
        return Integer.parseInt(chs.toString());
      }
    }
    return num;
  }

  /**
   * 最大数，把数组排成最大的数，排序，即贪心，类似参考「拼接最大数」
   *
   * <p>对 nums 按照 ab>ba 为 b>a，前导零
   *
   * <p>先单独证明两个数需要满足该定律，比如 3 & 30 有 330>303 显然 3 需要安排至 30 前，即表现为 3<30
   *
   * <p>再证明传递性，即两两之间都要满足该性质，参考
   * https://leetcode-cn.com/problems/largest-number/solution/gong-shui-san-xie-noxiang-xin-ke-xue-xi-vn86e/
   *
   * <p>扩展1，最小数 / 把数组排成最小的数，调整本题的排序规则为 ab>ba -> a>b 即可，参考
   * https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/
   *
   * <p>扩展2，参下 maxLessNumber
   *
   * @param nums
   * @return
   */
  public String largestNumber(int[] nums) {
    StringBuilder maxNum = new StringBuilder();
    List<String> strs = new ArrayList<>(nums.length);
    for (int n : nums) {
      strs.add(String.valueOf(n));
    }
    strs.sort((s1, s2) -> (s2 + s1).compareTo(s1 + s2));
    for (String n : strs) {
      maxNum.append(n);
    }
    return maxNum.toString();
    // 「最小数」需要去除前导零，因为可能有 02>20
    //    int start = 0;
    //    while (start < nums.length - 1 && res.charAt(start) == '0') {
    //      start += 1;
    //    }
    //    return res.substring(start);
  }

  /**
   * TODO 给定一个与一组正数，求由 A 中元素组成的小于 n 的最大数，如 {2,4,9} 小于 23121 的最大数为 22999
   *
   * <p>从最高位向最低位构造目标数，用 A 中尽量大的元素（但要小于等于 n 的相应位数字）。
   *
   * <p>一旦目标数中有一位数字小于 n 相应位的数字，剩余低位可用 A 中最大元素填充。
   *
   * <p>可能构造出等于 n 的数，需判断后重新构造。
   *
   * <p>若 A 中没有小于等于 n 最高位数字的元素，则可直接用 A 中最大元素填充低
   *
   * @param target
   * @param nums
   * @return
   */
  public int maxLessNumber(int target, int[] nums) {
    char[] digits = String.valueOf(target).toCharArray();
    Arrays.sort(nums);
    int[] resNums = new int[digits.length];
    int write = 0;
    for (char d : digits) {
      // 找到刚好小于 d 的数
      int lessNum = -1;
      int lo = 0, hi = nums.length - 1;
      while (lo < hi) {
        int mid = lo + (hi - lo) / 2, cur = nums[mid];
        if (mid > 0 && nums[mid - 1] < d && cur >= d) {
          lessNum = cur;
          break;
        }
        if (cur < d) lo = mid + 1;
        else hi = mid;
      }
      if (lessNum > -1) {
        resNums[write] = lessNum;
        resNums[write + 1] = nums[nums.length - 1];
      }
      write += 1;
    }
    // 比如 2 & {2} 无解，需要统计，且去除前导零。
    int num = 0;
    for (int n : resNums) {
      num += num * 10 + n;
    }
    return num;
  }

  /**
   * 字典序的第k小数字，找到 [1,n] 内，前序
   *
   * @param n the n
   * @param k the k
   * @return int int
   */
  public int findKthNumber(int n, int k) {
    int count = 1, prefix = 1; // 字典序最小即起点为 1，其前缀为 1
    while (count < k) {
      int curCount = count(prefix, n);
      if (curCount + count > k) { // 本层，往下层遍历，一直遍历到第 K 个推出循环
        prefix *= 10;
        count += 1;
      } else { // 去下个前缀，即相邻子树遍历
        prefix += 1;
        count += curCount;
      }
    }
    return prefix; // 退出循环时 cur==k 正好找到
  }

  // DFS prefix 为根的树，统计至 n 的个数
  private int count(int lo, int hi) {
    // 下一个前缀峰头，而且不断向下层遍历乘 10 可能会溢出
    long cur = lo, nxt = cur + 1;
    int count = 0;
    while (cur <= hi) { // 逐层
      count += Math.min(hi + 1, nxt) - cur;
      cur *= 10;
      nxt *= 10;
    }
    return count;
  }

  /**
   * 字典序排数，按字典序返回 [1,n] 所有整数，N 叉树遍历
   *
   * <p>参考
   * https://leetcode-cn.com/problems/lexicographical-numbers/solution/386-zi-dian-xu-pai-shu-o1-kong-jian-fu-z-aea2/
   *
   * @param n the n
   * @return list list
   */
  public List<Integer> lexicalOrder(int n) {
    List<Integer> res = new ArrayList<>();
    int num = 1;
    while (res.size() < n) {
      while (num <= n) { // 1.深度遍历
        res.add(num);
        num *= 10;
      }
      // 2.回溯，当前层子节点遍历完，或不存在节点(因为已经大于 n)，则返回上一层
      while (num % 10 == 9 || num > n) {
        num /= 10;
      }
      num += 1; // 3.根的下一个子节点
    }
    return res;
  }

  /**
   * 第k个排列，[1:n] 所有数字全排列按数字序第 k 小。
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/permutation-sequence/solution/hui-su-jian-zhi-python-dai-ma-java-dai-ma-by-liwei/
   *
   * @param n
   * @param k
   * @return
   */
  public String getPermutation(int n, int k) {
    // 阶乘即该节点叶节点总数
    final int[] factorials = new int[n + 1];
    factorials[0] = 1;
    for (int i = 1; i < n; i++) {
      factorials[i] = factorials[i - 1] * i;
    }
    boolean[] visited = new boolean[n + 1];
    StringBuilder permutation = new StringBuilder(n);
    for (int i = n - 1; i > -1; i--) {
      int cnt = factorials[i];
      for (int j = 1; j <= n; j++) {
        if (!visited[j] && cnt >= k) {
          visited[j] = true;
          permutation.append(j);
          break;
        }
        if (cnt < k) k -= cnt;
      }
    }
    return permutation.toString();
  }

  /**
   * 拼接最大数，两个无序正整数数组，共取 k 个拼接为数字，求该数最大的方案
   *
   * <p>TODO
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @param k the k
   * @return int [ ]
   */
  //  public int[] maxNumber(int[] nums1, int[] nums2, int k) {}
}

/** 提供一些数组的通用方法 */
abstract class DefaultArray {
  protected int smallmostBound(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      // 下一轮搜索区间是 [lo...mid] 因为小于一定不是解
      if (nums[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return nums[lo] == target ? lo : -1;
  }

  protected int bigmostBound(int[] nums, int target, int start) {
    int lo = start, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2 + 1; // 需要右开区间
      // 下一轮搜索区间是 [lo..mid - 1]
      if (nums[mid] <= target) lo = mid;
      else hi = mid - 1;
    }
    return nums[hi] == target ? hi : -1;
  }

  /**
   * 调换
   *
   * @param nums the nums
   * @param a the a
   * @param b the b
   */
  protected final void swap(int[] nums, int a, int b) {
    int tmp = nums[a];
    nums[a] = nums[b];
    nums[b] = tmp;
  }

  protected final void swap(char[] nums, int a, int b) {
    char tmp = nums[a];
    nums[a] = nums[b];
    nums[b] = tmp;
  }

  /**
   * 返回顺序首个 k 满足 nums[k] != nums[start]，否则返回 -1
   *
   * @param nums the nums
   * @param start the start
   * @param upper the upper
   * @return int int
   */
  protected final int nextIdx(int[] nums, int start, boolean upper) {
    int step = upper ? 1 : -1;
    for (int i = start + 1; i < nums.length && i > -1; i += step) {
      if (nums[i] == nums[start]) continue;
      return i;
    }
    return -1;
  }

  /**
   * Reverse.
   *
   * @param nums the nums
   * @param start the start
   * @param end the end
   */
  protected void reverse(int[] nums, int start, int end) {
    int lo = start, hi = end;
    while (lo < hi) {
      swap(nums, lo, hi);
      lo += 1;
      hi -= 1;
    }
  }
}
