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
 * <p>寻找 & 统计，二分参考 https://www.zhihu.com/question/36132386/answer/530313852
 *
 * <p>相加 & 相乘
 *
 * <p>删除
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
    for (int cur = m + n - 1; cur >= 0; cur--) {
      int n1 = curNum(nums1, p1), n2 = curNum(nums2, p2);
      int bigger = 0;
      if (n1 <= n2) {
        bigger = n1;
        p1 -= 1; // nextIdx()
      } else {
        bigger = n2;
        p2 -= 1; // nextIdx()
      }
      nums1[cur] = bigger;
    }
  }

  /**
   * 三数之和
   *
   * @param nums the nums
   * @return list list
   */
  public List<List<Integer>> threeSum(int[] nums) {
    final int target = 0, limit = 3;
    List<List<Integer>> res = new ArrayList<>();
    Arrays.sort(nums);
    for (int i = 0; i < nums.length - (limit - 1); i++) {
      int num = nums[i];
      if (num > target) break;
      for (List<Integer> cur : twoSum(nums, i, target - num)) {
        cur.add(num);
        res.add(cur);
      }
    }
    return res;
  }

  /**
   * 两数之和
   *
   * @param nums the nums
   * @param start the start
   * @param target the target
   * @return the list
   */
  public List<List<Integer>> twoSum(int[] nums, int start, int target) {
    List<List<Integer>> res = new ArrayList<>();
    int lo = start, hi = nums.length - 1;
    while (lo < hi) {
      int sum = nums[lo] + nums[hi];
      if (sum < target) lo += 1;
      else if (sum == target) res.add(Arrays.asList(nums[lo], nums[hi]));
      else hi -= 1;
      lo += 1;
    }
    return res;
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
    // 题设至少三位
    int res = nums[0] + nums[1] + nums[2];
    for (int i = 0; i < nums.length; i++) {
      int pivot = nums[i];
      int lo = i + 1, hi = nums.length - 1;
      while (lo < hi) {
        int sum = pivot + nums[lo] + nums[hi];
        res = Math.abs(target - sum) < Math.abs(target - res) ? sum : res;
        if (sum < target) lo += 1;
        else if (sum == target) return res;
        else hi -= 1;
      }
    }
    return res;
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
    // 使用 left+1<right 防止只能将数组分成两个部分
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
   * 四数之和
   *
   * <p>TODO
   *
   * @param nums
   * @param target
   * @return
   */
  //  public List<List<Integer>> fourSum(int[] nums, int target) {}

  /**
   * 两个数组的交集，重复 & 顺序
   *
   * <p>扩展，有序则双指针 & 二分，否则对较小的数组建立 Hash 再遍历较大者
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return int [ ]
   */
  public int[] intersection(int[] nums1, int[] nums2) {
    return intersection1(nums1, nums2);
  }

  private int[] intersection1(int[] nums1, int[] nums2) {
    Arrays.sort(nums1);
    Arrays.sort(nums2);
    int len1 = nums1.length, len2 = nums2.length;
    int[] res = new int[len1 + len2];
    int cur = 0, p1 = 0, p2 = 0;
    while (p1 < len1 && p2 < len2) {
      int num1 = nums1[p1], num2 = nums2[p2];
      if (num1 < num2) p1 += 1;
      else if (num1 == num2) {
        // 保证加入元素的唯一性
        if (cur == 0 || num1 != res[cur - 1]) {
          res[cur] = num1;
          cur += 1;
        }
        p1 += 1;
        p2 += 1;
      } else p2 += 1;
    }
    return Arrays.copyOfRange(res, 0, cur);
  }

  private int[] intersection2(int[] a, int[] b) {
    int[] nums1 = (a.length < b.length) ? a : b, nums2 = (a.length < b.length) ? b : a;
    Map<Integer, Boolean> map = new HashMap<>();
    int cur = 0;
    int[] res = new int[nums1.length];
    for (int num : nums1) map.put(num, true);
    for (int num : nums2) {
      if (!map.containsKey(num) || !map.get(num)) continue;
      res[cur] = num;
      cur += 1;
      map.put(num, false);
    }
    return Arrays.copyOfRange(res, 0, cur);
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

/** 堆排序相关，建堆 */
class HHeap extends DefaultArray {
  /**
   * 数组中的第k个最大元素，原地维护小根堆
   *
   * <p>堆化 [0,k] & 依次入堆 [k+1,l-1] 的元素 & 最终堆顶即 [0]
   *
   * <p>扩展1，寻找两个有序数组的第 k 大，参下「寻找两个有序数组的中位数」
   *
   * <p>TODO 扩展2，判断 num 是否为第 k 大，有重复，partition 如果在 K 位置的左边和右边都遇到该数，直接结束，否则直到找到第 k 大的元素比较是否为同一个数
   *
   * <p>扩展3，如何只选出 [n, m]，分别建两个长度为 n & m-n 的小根堆，优先入前者，前者出队至入后者，后者不允则舍弃
   *
   * <p>扩展4，不同的量级如何选择，如 10 & 100 & 10k，分别为计数排序，快速选择 or 建堆，分治 & 外排
   *
   * @param nums the nums
   * @param k the k
   * @return the int
   */
  public int findKthLargest(int[] nums, int k) {
    // 对前 k 个元素建小根堆，即堆化 [0,k-1] 区间的元素
    for (int i = 0; i < k; i++) {
      swim(nums, i);
    }
    // 剩下的元素与堆顶比较，若大于堆顶则去掉堆顶，再将其插入
    for (int i = k; i < nums.length; i++) {
      if (priorityThan(nums[i], nums[0])) continue;
      swap(nums, 0, i);
      sink(nums, 0, k - 1);
    }
    //    int idx = nums.length - 1;
    //    heapify(nums, nums.length);
    //    while (idx >= nums.length - k + 1) {
    //      swap(nums, 0, idx);
    //      idx--;
    //      down(nums, 0, idx - 1);
    //    }
    // 结束后第 k 个大的数即小根堆堆顶
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
    int idx = nums.length - 1;
    heapify(nums, nums.length);
    // 循环不变量，区间 [0, idx] 堆有序
    while (idx >= 1) {
      // 把堆顶元素（当前最大）交换到数组末尾
      swap(nums, 0, idx);
      // 逐步减少堆有序的部分
      idx -= 1;
      // 下标 0 位置下沉操作，使得区间 [0, i] 堆有序
      sink(nums, 0, idx);
    }
  }

  // 从 (len-1)/2 即首个叶结点开始逐层下沉
  private void heapify(int[] nums, int capcacity) {
    for (int i = capcacity / 2; i >= 0; i--) {
      sink(nums, i, capcacity - 1);
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

  // 从下到上调整堆，分别取左右子结点 2*cur+1 与 +2 判断
  private void sink(int[] heap, int idx, int end) {
    int cur = idx, child = 2 * cur + 1;
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

  // v1 是否优先度高于 v2
  private boolean priorityThan(int v1, int v2) {
    return v1 < v2;
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
      if (nums[i] > max) {
        second = max;
        max = nums[i];
      } else {
        second = Math.max(second, nums[i]);
      }
    }
    return new int[] {max, second};
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
    int[][] res = new int[K][2];
    PriorityQueue<int[]> pq =
        new PriorityQueue<>(
            K, (p1, p2) -> p2[0] * p2[0] + p2[1] * p2[1] - p1[0] * p1[0] - p1[1] * p1[1]);
    for (int[] point : points) {
      if (pq.size() < K) {
        pq.offer(point);
        continue;
      }
      // 判断当前点的距离是否小于堆中的最大距离
      if (pq.comparator().compare(point, pq.peek()) > 0) {
        pq.poll();
        pq.offer(point);
      }
    }
    for (int i = 0; i < K; i++) {
      res[i] = pq.poll();
    }
    return res;
  }
}

/**
 * 基于比较的排序的时间复杂度下界均是 nlogn
 *
 * <p>数组全排列共有 n! 种情况，而二分每次最多能排除一半的情况，根据斯特林级数，算法的渐进复杂度为 O(log(n!)) = O(nlogn)
 *
 * <p>链表快排参考「排序链表」
 *
 * <p>TODO 快排最优 & 平均 & 最坏的复杂度分别如何计算
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
    if (lo >= hi) return;
    int pivotIdx = lo + random.nextInt(hi - lo + 1), pivot = nums[pivotIdx];
    // 下方需要确保虚拟头也满足 <pivot，因此从 lt+1 开始遍历
    swap(nums, pivotIdx, lo);
    // 虚拟头尾，保证界外
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
    if (nums.length < 2) return;
    int pivot = 1;
    int lt = -1, gt = nums.length; // 虚拟头尾，保证界外
    int cur = 0;
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

class MMerge extends DefaultArray {
  private int res = 0;
  /**
   * 归并排序，up-to-bottom 递归，先分后合
   *
   * <p>bottom-to-up 迭代，参考排序链表
   *
   * @param nums the nums
   * @param lo the lo
   * @param hi the hi
   */
  public void mergeSort(int[] nums, int lo, int hi) {
    if (nums.length < 2) return;
    divide1(nums, new int[nums.length], 0, nums.length - 1);
  }

  // 取中二分 & 合
  private void divide1(int[] nums, int[] tmp, int lo, int hi) {
    // 对于双指针，假如迭代内部基于比较，则不需要=，假如需要统计每个元素，则需要
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    divide1(nums, tmp, lo, mid);
    divide1(nums, tmp, mid + 1, hi);
    // curing 因为此时 [lo,mid]&[mid+1,hi] 分别有序，否则说明二者在数轴上范围存在重叠
    if (nums[mid] <= nums[mid + 1]) {
      return;
    }
    merge1(nums, tmp, lo, mid, hi); // 区间两两相邻合并
  }

  // 合并 nums[lo:mid] & nums[mid+1:hi] 即排序区间 [lo,hi]
  // 四种情况，其一遍历结束 & 比较
  // 写成 < 会丢失稳定性，因为相同元素原来靠前的排序以后依然靠前，因此排序稳定性的保证必需 <=
  private void merge1(int[] nums, int[] tmp, int lo, int mid, int hi) {
    // 前后指针未逾界，且数组至少有两个元素
    if (hi - lo >= 1) {
      System.arraycopy(nums, lo, tmp, lo, hi + 1 - lo);
    }
    int p1 = lo, p2 = mid + 1;
    for (int i = lo; i <= hi; i++) {
      if (p1 == mid + 1) {
        nums[i] = tmp[p2];
        p2 += 1;
      } else if (p2 == hi + 1) {
        nums[i] = tmp[p1];
        p1 += 1;
      } else if (tmp[p1] <= tmp[p2]) {
        nums[i] = tmp[p1];
        p1 += 1;
      } else if (tmp[p1] > tmp[p2]) {
        nums[i] = tmp[p2];
        p2 += 1;
      }
    }
  }

  /**
   * 数组中的逆序对，与上方「归并排序」基本一致
   *
   * @param nums the nums
   * @return int int
   */
  public int reversePairs(int[] nums) {
    if (nums.length < 2) return 0;
    divide2(nums, new int[nums.length], 0, nums.length - 1);
    return res;
  }

  private void divide2(int[] nums, int[] tmp, int lo, int hi) {
    if (lo == hi) return;
    int mid = lo + (hi - lo) / 2;
    divide2(nums, tmp, lo, mid);
    divide2(nums, tmp, mid + 1, hi);
    if (nums[mid] <= nums[mid + 1]) {
      return;
    }
    res += mergeAndCount(nums, tmp, lo, mid, hi);
  }

  private int mergeAndCount(int[] nums, int[] tmp, int lo, int mid, int hi) {
    int cur = 0;
    if (hi - lo + 1 >= 0) {
      System.arraycopy(nums, lo, tmp, lo, hi - lo + 1);
    }
    int p1 = lo, p2 = mid + 1;
    for (int i = lo; i <= hi; i++) {
      if (p1 == mid + 1) {
        nums[i] = tmp[p2];
        p2 += 1;
      } else if (p2 == hi + 1) {
        nums[i] = tmp[p1];
        p1 += 1;
      } else if (tmp[p1] <= tmp[p2]) {
        nums[i] = tmp[p1];
        p1 += 1;
      } else if (tmp[p1] > tmp[p2]) {
        nums[i] = tmp[p2];
        p2 += 1;
        cur += mid - p1 + 1;
      }
    }
    return cur;
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
}

/**
 * 二分，参考上方 AArray.search 的写法
 *
 * <p>lo<=hi 明确碰撞的含义
 */
class Dichotomy extends DefaultArray {
  /**
   * 寻找两个有序数组的中位数，单数组已去重，相互之间可能有重，联合对两个数组二分求 topk 则复杂度为 log(m+n)
   *
   * <p>对于两个有序数组，逆序则顺序遍历求第 k 大，正序则顺序遍历求第 k 小，反之，二者均需要逆序遍历
   *
   * <p>对于 123 & 334 的中位数和 top3 分别是 3 & 2
   *
   * <p>扩展1，无序数组找中位数，建小根堆 len/2+1 奇数则堆顶，否则出队一次 & 堆顶取平均，海量数据参考「数据流的中位数」
   *
   * <p>扩展2，单纯求两个逆序数组 topk，则双指针从尾开始遍历即可，且需要对值，而本题求中位数本质是对排位进行二分，前提是单个数组无重复
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return double double
   */
  public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int n = nums1.length, m = nums2.length;
    int ranking = (n + m + 1) / 2, rankingMore = (n + m + 2) / 2;
    // 将偶数和奇数的情况合并，奇数则求两次同样的 k
    //    return (getkSmallElement2(nums1, 0, n - 1, nums2, 0, m - 1, ranking)
    //            + getkSmallElement2(nums1, 0, n - 1, nums2, 0, m - 1, rankingMore))
    //        * 0.5;
    return (getKSmallElement1(nums1, nums2, ranking) + getKSmallElement1(nums1, nums2, rankingMore))
        * 0.5;
  }

  // 迭代，特判其一为空 & 其一遍历结束 & k 为 1
  private int getKSmallElement1(int[] nums1, int[] nums2, int k) {
    int l1 = nums1.length, l2 = nums2.length;
    int p1 = 0, p2 = 0, curRanking = k;
    while (true) {
      // 去重1
      if (p1 == l1) {
        return nums2[p2 + curRanking - 1];
      } else if (p2 == l2) {
        return nums1[p1 + curRanking - 1];
      } else if (curRanking == 1) {
        return Math.min(nums1[p1], nums2[p2]);
      }
      int half = curRanking / 2;
      int newIdx1 = Math.min(p1 + half, l1) - 1, newIdx2 = Math.min(p2 + half, l2) - 1;
      int num1 = nums1[newIdx1], num2 = nums2[newIdx2];
      // 去重2
      if (num1 <= num2) {
        curRanking -= (newIdx1 - p1 + 1);
        p1 = newIdx1 + 1;
      } else {
        curRanking -= (newIdx2 - p2 + 1);
        p2 = newIdx2 + 1;
      }
    }
  }

  // 尾递归
  // 特判 k=1 & 其一为空
  private int getkSmallElement2(
      int[] nums1, int lo1, int hi1, int[] nums2, int lo2, int hi2, int k) {
    if (k == 1) return Math.min(nums1[lo1], nums2[lo2]);
    int len1 = hi1 - lo1 + 1, len2 = hi2 - lo2 + 1;
    // 让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1
    if (len1 > len2) {
      return getkSmallElement2(nums2, lo2, hi2, nums1, lo1, hi1, k);
    } else if (len1 == 0) {
      return nums2[lo2 + k - 1];
    }
    int p1 = lo1 + Math.min(len1, k / 2) - 1, p2 = lo2 + Math.min(len2, k / 2) - 1;
    return (nums1[p1] > nums2[p2])
        ? getkSmallElement2(nums1, lo1, hi1, nums2, p2 + 1, hi2, k - (p2 - lo2 + 1))
        : getkSmallElement2(nums1, p1 + 1, hi1, nums2, lo2, hi2, k - (p1 - lo1 + 1));
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
    int[] res = new int[] {-1, -1};
    if (nums.length < 1 || nums[0] > target || nums[nums.length - 1] < target) {
      return res;
    }
    int lower = lowerBound(nums, target);
    if (lower == -1) {
      return res;
    }
    int upper = upperBound(nums, target, lower);
    return new int[] {lower, upper};
  }

  protected int lowerBound(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      // 下一轮搜索区间是 [lo...mid] 因为小于一定不是解
      if (nums[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    return nums[lo] == target ? lo : -1;
  }

  private int upperBound(int[] nums, int target, int start) {
    int lo = start, hi = nums.length - 1;
    while (lo < hi) {
      // 需要右开区间
      int mid = lo + (hi - lo) / 2 + 1;
      // 下一轮搜索区间是 [lo..mid - 1]
      if (nums[mid] <= target) lo = mid;
      else hi = mid - 1;
    }
    return nums[hi] == target ? hi : -1;
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
      // 根据分支的逻辑将中间数改成上取整
      int mid = lo + (hi - lo + 1) / 2;
      if (nums[mid] < nums[hi]) {
        // target 落在 [mid..right]
        if (nums[mid] <= target && target <= nums[hi]) lo = mid;
        else hi = mid - 1;
      } else {
        // target 落在 [left..mid - 1]
        if (nums[lo] <= target && target <= nums[mid - 1]) hi = mid - 1;
        else lo = mid;
      }
    }
    return nums[lo] == target ? lo : -1;
  }

  /**
   * 寻找旋转排序数组中的最小值，比较边界
   *
   * <p>扩展1，找最大，可复用本题，参考「山脉数组的顶峰索引」
   *
   * <p>扩展2，存在重复元素，参下
   *
   * <p>扩展3，降序且旋转的数组，求最小值
   *
   * @param nums the nums
   * @return int int
   */
  public int findMin(int[] nums) {
    return findMinI(nums);
  }

  // 已去重
  private int findMinI(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (nums[mid] < nums[hi]) hi = mid;
      else lo = mid + 1;
    }
    return nums[lo];
  }

  // 寻找旋转排序数组中的最小值II，有重复
  private int findMinII(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (nums[mid] < nums[hi]) hi = mid;
      else if (nums[mid] > nums[hi]) lo = mid + 1;
      else hi -= 1; // 比上方多的一个判断
    }
    return nums[lo];
  }

  /**
   * 寻找峰值，比较相邻
   *
   * <p>对比其余二分，此处需要 lo < hi
   *
   * @param nums the nums
   * @return int int
   */
  public int findPeakElement(int[] nums) {
    int lo = 0, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (nums[mid] < nums[mid + 1]) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /**
   * 山脉数组的顶峰索引，与上方几乎一致
   *
   * <p>TODO
   *
   * @param nums
   * @return
   */
  public int peakIndexInMountainArray(int[] nums) {
    int lo = 1, hi = nums.length - 2;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      // 缩减区间为 [mid+1,hi]
      if (nums[mid] < nums[mid + 1]) lo = mid + 1;
      else hi = mid;
    }
    // 碰撞时结束
    return lo;
  }

  /**
   * 山脉数组中查找目标值
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
    int peak = lo, idx = search(mountainArr, target, 0, peak, true);
    return idx != -1 ? idx : search(mountainArr, target, peak + 1, mountainArr.length() - 1, false);
  }

  private int search(MountainArray mountainArr, int target, int lo, int hi, boolean flag) {
    if (!flag) target *= -1;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      int cur = mountainArr.get(mid) * (flag ? 1 : -1);
      if (cur < target) {
        lo = mid + 1;
      } else if (cur == target) {
        return mid;
      } else {
        hi = mid;
      }
    }
    int mid = lo + (hi - lo) / 2;
    int cur = mountainArr.get(mid) * (flag ? 1 : -1);
    if (cur == target) return lo;
    return -1;
  }

  /**
   * 搜索二维矩阵，建议模拟 BST 以右上角作根开始遍历，复杂度 m+n
   *
   * <p>I & II 通用，或二分，复杂度为 mlogn
   *
   * @param matrix the matrix
   * @param target the target
   * @return boolean boolean
   */
  public boolean searchMatrix(int[][] matrix, int target) {
    int y = 0, x = matrix[0].length - 1;
    while (y < matrix.length && x >= 0) {
      if (matrix[y][x] < target) y += 1;
      else if (matrix[y][x] == target) return true;
      else if (matrix[y][x] > target) x -= 1;
    }
    return false;
  }

  /**
   * 有序矩阵中第k小的元素，值域二分
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/solution/er-fen-chao-ji-jian-dan-by-jacksu1024/
   *
   * @param matrix
   * @param k
   * @return
   */
  public int kthSmallest(int[][] matrix, int k) {
    int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1];
    while (lo < hi) {
      // 每次循环都保证第 k 小的数在 [lo,hi]
      int mid = lo + (hi - lo) / 2;
      // 找二维矩阵中 <=mid 的元素总个数，判断目标分别在 [lo,mid] 或 [mid+1,hi]
      if (findLteMid(matrix, mid) < k) lo = mid + 1;
      else hi = mid;
    }
    return hi;
  }

  // 向右下角遍历，找每列最后一个 <=mid 的数即知道每一列有多少个数 <=mid
  private int findLteMid(int[][] matrix, int mid) {
    int count = 0;
    int y = matrix.length - 1, x = 0;
    while (-1 < y && x < matrix[0].length) {
      if (matrix[y][x] <= mid) {
        // 第 j 列有 i+1 个元素 <=mid
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
   * 有序数组中的单一元素，下述代码适配无序
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
      if (mid % 2 == 0 && nums[mid] == nums[mid + 1]) {
        lo = mid + 2;
      } else if (mid % 2 == 1 && nums[mid] == nums[mid - 1]) {
        // 同上，但如 11445 5 66899
        lo = mid + 1;
      } else {
        // 取中心值时会向下取整，如 11455 6 68899 & 1145 5 6688
        hi = mid;
      }
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
    List<int[]> res = new ArrayList<int[]>();
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
        res.add(ans);
        lo += 1;
      } else {
        lo += 1;
      }
    }
    return res.toArray(new int[res.size()][]);
  }

  /**
   * 数据流的中位数，分别使用两个堆并保证二者元素数目差值不超过 2 即可
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/find-median-from-data-stream/solution/gong-shui-san-xie-jing-dian-shu-ju-jie-g-pqy8/
   */
  //  class MedianFinder {
  //
  //    public MedianFinder() {}
  //
  //    public void addNum(int num) {}
  //
  //    public double findMedian() {}
  //  }
}

/** 前缀和，区间和满足 target */
class PreSum {
  /**
   * 最大子数组和 / 最大子序和 / 连续子数组的最大和，基于贪心，通过前缀和
   *
   * <p>dp[i] 表示以 nums[i] 结尾的最大子序和，状态压缩为 curSum
   *
   * <p>sum>0 说明 sum 对结果有增益效果，则后者保留并加上当前遍历数字，否则舍弃，sum 直接更新为当前遍历数字
   *
   * <p>扩展1，要求返回子数组，则添加始末指针，每当 curSum<=0 时更新
   *
   * <p>扩展2，返回最大和的子序列
   *
   * @param nums the nums
   * @return int int
   */
  public int maxSubArray(int[] nums) {
    int curSum = 0, res = nums[0];
    for (int num : nums) {
      curSum = curSum > 0 ? curSum + num : num;
      res = Math.max(res, curSum);
    }
    return res;
  }

  /**
   * 和为k的子数组
   *
   * <p>参考
   * https://leetcode-cn.com/problems/subarray-sum-equals-k/solution/de-liao-yi-wen-jiang-qian-zhui-he-an-pai-yhyf/
   *
   * <p>扩展1，求和为 k 的子数组中，最大的长度，参考「连续数组」
   *
   * @param nums the nums
   * @param k the k
   * @return int int
   */
  public int subarraySum(int[] nums, int k) {
    int count = 0, preSum = 0;
    // 对应的前缀和的个数
    HashMap<Integer, Integer> idxBySum = new HashMap<>();
    // 需要预存前缀和为 0，会漏掉前几位就满足的情况
    // 如 [1,1,0]，k=2 如果没有这行代码，则会返回 0 漏掉 1+1=2 与 1+1+0=2
    // 输入 [3,1,1,0] k=2 时则不会漏掉，因为 presum[3]-presum[0] 表示前面 3 位的和
    idxBySum.put(0, 1);
    for (int i = 0; i < nums.length; i++) {
      preSum += nums[i];
      // 当前前缀和已知，判断是否含有 presum-k 的前缀和，那么我们就知道某一区间的和为 k
      if (idxBySum.containsKey(preSum - k)) {
        count += idxBySum.get(preSum - k);
      }
      idxBySum.put(preSum, idxBySum.getOrDefault(preSum, 0) + 1);
    }
    return count;
  }

  /**
   * 连续的子数组和，返回是否存在子数组满足总和为 k 的倍数，且至少有两个元素
   *
   * <p>TODO 前缀和
   *
   * <p>扩展1，k 倍区间方案数，参考
   * https://leetcode-cn.com/problems/continuous-subarray-sum/solution/gong-shui-san-xie-tuo-zhan-wei-qiu-fang-1juse/
   *
   * @param nums the nums
   * @param k the k
   * @return boolean
   */
  public boolean checkSubarraySum(int[] nums, int k) {
    HashMap<Integer, Integer> idxBySum = new HashMap<>();
    idxBySum.put(0, -1);
    int preSum = 0;
    for (int i = 0; i < nums.length; ++i) {
      preSum += nums[i];
      int key = k == 0 ? preSum : preSum % k;
      if (idxBySum.containsKey(key)) {
        if (i - idxBySum.get(key) >= 2) return true;
        // 因为我们需要保存最小索引，当已经存在时则不用再次存入，不然会更新索引值
        continue;
      }
      idxBySum.put(key, i);
    }
    return false;
  }

  /**
   * 连续数组，找到数量相同的 0 和 1 的最长子数组，题设非 0 即 1
   *
   * <p>将 0 作为 −1，则转换为求区间和满足 0 的最长子数组，同时记录「某个前缀和出现的最小下标」
   *
   * <p>参考
   * https://leetcode-cn.com/problems/contiguous-array/solution/qian-zhui-he-chai-fen-ha-xi-biao-java-by-liweiwei1/
   *
   * @param nums the nums
   * @return int
   */
  public int findMaxLength(int[] nums) {
    int preSum = 0, maxLen = 0;
    // 总和及其首次出现的下标
    Map<Integer, Integer> idxBySum = new HashMap<>();
    // 可能存在前缀和刚好满足条件的情况
    idxBySum.put(0, -1);
    for (int i = 0; i < nums.length; i++) {
      preSum += nums[i] == 0 ? -1 : 1;
      // 因为求的是最长的长度，只记录前缀和第一次出现的下标
      if (idxBySum.containsKey(preSum)) maxLen = Math.max(maxLen, i - idxBySum.get(preSum));
      else idxBySum.put(preSum, i);
    }
    return maxLen;
  }

  /**
   * 长度最小的子数组，满足和不少于 target，前缀和搭配滑窗的思维
   *
   * <p>扩展1，列出所有满足和为 target 的连续子序列，参考「和为k的子数组」，参下 annotate
   *
   * <p>扩展2，里边有负数，参考「和至少为k的最短子数组」
   *
   * @param target the target
   * @param nums the nums
   * @return int int
   */
  public int minSubArrayLen(int target, int[] nums) {
    int minLen = Integer.MAX_VALUE, preSum = 0;
    int lo = 0, hi = 0;
    while (hi < nums.length) {
      // in
      preSum += nums[hi];
      while (preSum >= target) {
        minLen = Math.min(minLen, hi - lo + 1);
        // out
        preSum -= nums[lo];
        lo += 1;
      }
      // 替换上一段 while
      //      if (sum == target) {
      //        // 入结果集
      //        hi += 1;
      //        continue;
      //      }
      //      while (sum > target) {
      //        sum -= nums[lo];
      //        if (sum == target) {
      //          // 入结果集
      //        }
      //        lo += 1;
      //      }
      hi += 1;
    }
    return minLen == Integer.MAX_VALUE ? 0 : minLen;
  }

  /**
   * 和至少为k的最短子数组，单调队列 & 前缀和
   *
   * <p>TODO
   *
   * <p>需要找到索引 x & y 使得 prefix[y]-prefix[x]>=k 且 y-x 最小
   *
   * @param nums the nums
   * @param k the k
   * @return int int
   */
  public int shortestSubarray(int[] nums, int k) {
    int len = nums.length;
    long[] prefix = new long[len + 1];
    for (int i = 0; i < len; i++) {
      prefix[i + 1] = prefix[i] + (long) nums[i];
    }
    // len+1 is impossible
    int res = len + 1;
    // 单调队列
    Deque<Integer> mq = new LinkedList<>();
    for (int i = 0; i < prefix.length; i++) {
      // Want opt(y) = largest x with prefix[x]<=prefix[y]-K
      while (!mq.isEmpty() && prefix[i] <= prefix[mq.getLast()]) {
        mq.removeLast();
      }
      while (!mq.isEmpty() && prefix[i] >= prefix[mq.getFirst()] + k) {
        res = Math.min(res, i - mq.removeFirst());
      }
      mq.addLast(i);
    }
    return res < len + 1 ? res : -1;
  }
}

/** 遍历相关 */
class Travesal extends DefaultArray {
  /**
   * 寻找重复数，无序找首个，快慢指针
   *
   * <p>参考
   * https://leetcode-cn.com/problems/find-the-duplicate-number/solution/kuai-man-zhi-zhen-de-jie-shi-cong-damien_undoxie-d/
   *
   * <p>TODO 扩展1，重复数字有多个，要求找出所有重复数字，复杂度为 n & 1，nums[i] 每出现过一次对 nums[idx]+=n，其中 idx=nums[i]-1，加完之后，当
   * nums[idx]>2*n 时就能表示 nums[i]，即 idx+1 出现过两次
   *
   * @param nums the nums
   * @return int int
   */
  public int findDuplicate(int[] nums) {
    int lo = 0, hi = 0;
    while (true) {
      lo = nums[lo];
      hi = nums[nums[hi]];
      if (hi == lo) break;
    }
    int finder = 0;
    while (true) {
      finder = nums[finder];
      lo = nums[lo];
      if (lo == finder) break;
    }
    return lo;
  }

  // 对有序数组找到重复数超过 k 的序列，滑窗，双指针间距超过 k-1 即可
  // 找到最接近 k 的下界索引 & 从此开始按照上述模板
  //  private int[] findDuplicatesK(int[] nums, int k) {
  //    List<Integer> res = new ArrayList<>();
  //    int lo = 0, hi = nums.length - 1;
  //    if (nums[hi] < k) return new int[] {};
  //    while (lo < hi) {
  //      int mid = lo + (hi - lo) / 2;
  //      if (nums[mid] < k) {
  //
  //      } else if (nums[mid] == k) {
  //
  //      } else if (nums[mid] > k) {
  //
  //      }
  //    }
  //
  //    int lo = 0;
  //    for (int hi = 1; hi < nums.length; hi++) {
  //      if (nums[hi] == nums[lo]) continue;
  //      if (hi - lo >= k) {
  //        res.add(nums[lo]);
  //      }
  //      lo = hi;
  //    }
  //    return res.stream().mapToInt(i -> i).toArray();
  //  }

  /**
   * 数组中重复的数据，题设每个数字至多出现两次，且在 [1,n] 内
   *
   * <p>原地哈希，重复会命中同一索引，nums[nums[i]-1]*=-1，类似缺失的第一个整数
   *
   * @param nums the nums
   * @return list list
   */
  public List<Integer> findDuplicates(int[] nums) {
    List<Integer> res = new ArrayList<>();
    for (int num : nums) {
      num *= num < 0 ? -1 : 1;
      int idx = num - 1;
      if (nums[idx] < 0) res.add(num);
      else nums[idx] *= -1;
    }
    return res;
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
    for (int i = 0; i < nums.length; i++) {
      // 不断判断 i 位置上被放入正确的数，即 nums[i]-1
      while (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]) {
        swap(nums, nums[i] - 1, i);
      }
    }
    for (int i = 0; i < nums.length; i++) {
      if (nums[i] != i + 1) return i + 1;
    }
    return nums.length + 1;
  }

  /**
   * 旋转数组，反转三次，all & [0,k-1] & [k,end]
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
   * 旋转图像
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
        int tmp = matrix[y][x];
        matrix[y][x] = matrix[x][y];
        matrix[x][y] = tmp;
      }
    }
    for (int i = 0; i < len; i++) {
      int[] curRow = matrix[i];
      int lo = 0, hi = len - 1;
      while (lo <= hi) {
        swap(curRow, lo, hi);
        lo += 1;
        hi -= 1;
      }
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
    if (matrix.length == 0) return res;
    int up = 0, down = row - 1, left = 0, right = col - 1;
    while (true) {
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
   * 字符的最短距离，依次正序和逆序遍历
   *
   * <p>TODO
   *
   * @param s
   * @param c
   * @return
   */
  public int[] shortestToChar(String s, char c) {
    int len = s.length(), pre = Integer.MIN_VALUE / 2;
    int[] res = new int[len];
    for (int i = 0; i < len; i++) {
      if (s.charAt(i) == c) pre = i;
      res[i] = i - pre;
    }
    pre = Integer.MAX_VALUE / 2;
    for (int i = len - 1; i >= 0; i--) {
      if (s.charAt(i) == c) pre = i;
      res[i] = Math.min(res[i], pre - i);
    }
    return res;
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

  /**
   * 分割数组的最大值
   *
   * <p>TODO
   *
   * @param nums
   * @param m
   * @return
   */
  public int splitArray(int[] nums, int m) {
    int max = 0, sum = 0;
    // 计算子数组各自和的最大值的上下界
    for (int num : nums) {
      max = Math.max(max, num);
      sum += num;
    }
    // 二分确定一个恰当的子数组各自的和的最大值，使得它对应的「子数组的分割数」恰好等于 m
    int lo = max, hi = sum;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      int splits = split(nums, mid);
      if (splits > m) {
        // 如果分割数太多，说明「子数组各自的和的最大值」太小，此时需要将「子数组各自的和的最大值」调大
        // 下一轮搜索的区间是 [mid + 1, right]
        lo = mid + 1;
      } else {
        // 下一轮搜索的区间是上一轮的反面区间 [left, mid]
        hi = mid;
      }
    }
    return lo;
  }

  /***
   *
   * @param nums 原始数组
   * @param maxIntervalSum 子数组各自的和的最大值
   * @return 满足不超过「子数组各自的和的最大值」的分割数
   */
  private int split(int[] nums, int maxIntervalSum) {
    // 至少是一个分割 & 当前区间的和
    int splits = 1, curIntervalSum = 0;
    for (int num : nums) {
      // 尝试加上当前遍历的这个数，如果加上去超过了「子数组各自的和的最大值」，就不加这个数，另起炉灶
      if (curIntervalSum + num > maxIntervalSum) {
        curIntervalSum = 0;
        splits += 1;
      }
      curIntervalSum += num;
    }
    return splits;
  }
}

/** TODO 移除相关，类似滑窗，后期考虑同步后者的模板 */
class Delete extends DefaultArray {
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
    final int target = 0; // diff 1
    int last = 0, k = 0;
    for (int hi = 0; hi < nums.length; hi++) {
      //      if (nums[hi] != 1 && nums[hi] != 6 && nums[hi] != 3) {
      if (last >= k && target == nums[hi]) {
        continue;
      }
      swap(nums, last, hi); // diff 2
      last += 1;
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
    char[] res = str.toCharArray();
    int last = 0;
    for (int hi = 0; hi < str.length(); hi++) {
      if (target == str.charAt(hi)) {
        continue;
      }
      res[last] = str.charAt(hi);
      last += 1;
    }
    return String.valueOf(Arrays.copyOfRange(res, 0, last));
  }

  /**
   * 删除排序数组中的重复项，保留 k 位
   *
   * <p>遇到目标则跳过
   *
   * <p>扩展1，参考删除字符串中的所有相邻重复项
   *
   * @param nums the nums
   * @return the int
   */
  public int removeDuplicates(int[] nums) {
    return removeDuplicatesI(nums);
  }

  /**
   * 删除排序数组中的重复项I，每个元素至多出现一次
   *
   * <p>原地，解法等同移动零，需要移除的目标位 nums[last - k]
   *
   * @param nums the nums
   * @return int the lo
   */
  private int removeDuplicatesI(int[] nums) {
    // final int target = nums[last - k];
    int last = 0, k = 1;
    for (int num : nums) {
      if (last >= k && nums[last - k] == num) continue;
      nums[last] = num;
      last += 1;
    }
    return last;
  }

  /**
   * 删除排序数组中的重复项II，每个元素至多出现两次
   *
   * <p>原地，解法等同移动零，需要移除的目标位 nums[last - k]
   *
   * @param nums the nums
   * @return the int
   */
  private int removeDuplicatesII(int[] nums) {
    // final int target = nums[last - k];
    int last = 0, k = 2;
    for (int num : nums) {
      if (last >= k && nums[last - k] == num) continue;
      nums[last] = num;
      last += 1;
    }
    return last;
  }

  /**
   * 删除字符串中的所有相邻重复项，毫无保留，原地建栈或模拟栈
   *
   * <p>类似有效的括号，即括号匹配，通过 top 指针模拟栈顶，即原地栈，且修改源数组
   *
   * <p>匹配指当前字符与栈顶不同，即入栈，否则出栈，且 skip 当前 char
   *
   * <p>最终栈内即为最终结果
   *
   * @param s the s
   * @return string string
   */
  public String removeDuplicates(String s) {
    StringBuilder stack = new StringBuilder();
    int nextTop = -1;
    for (int i = 0; i < s.length(); ++i) {
      char ch = s.charAt(i);
      if (nextTop >= 0 && stack.charAt(nextTop) == ch) {
        stack.deleteCharAt(nextTop);
        nextTop -= 1;
      } else {
        stack.append(ch);
        nextTop += 1;
      }
    }
    return stack.toString();
  }

  /**
   * 调整数组顺序使奇数位于偶数前面，参考移动零，即遇到目标则跳过
   *
   * <p>扩展1，链表参考「奇偶链表」
   *
   * @param nums
   * @return
   */
  public int[] exchange(int[] nums) {
    int lo = 0;
    for (int hi = 0; hi < nums.length; hi++) {
      if ((nums[hi] & 1) == 0) continue;
      swap(nums, lo, hi);
      lo += 1;
    }
    return nums;
  }
}

/** 字典序相关 */
class DicOrder extends DefaultArray {
  /**
   * 下一个排列，求按照字典序，该排列下一个大的
   *
   * <p>对比下方「最大交换」，后者是找到交换结果的最大
   *
   * <p>找 & 排 & 找 & 换 & 排
   *
   * <p>扩展1，上一个排列，从 n-2 开始找到首个峰 & 峰右边调为降序 & 从 n-1 开始找到首个比峰小的数，交换
   *
   * @param nums the nums
   */
  public void nextPermutation(int[] nums) {
    // for (int i = nums.length - 2; i > 0; i--) {
    for (int i = nums.length - 1; i > 0; i--) {
      if (nums[i] <= nums[i - 1]) continue;
      // 左闭右开
      Arrays.sort(nums, i, nums.length);
      for (int j = i; j < nums.length; j++) {
        if (nums[j] <= nums[i - 1]) continue;
        swap(nums, i - 1, j);
        return;
      }
    }
    Arrays.sort(nums);
  }

  /**
   * 最大交换，只交换一次任意两位的数，使其结果数值是所有方案中最大的
   *
   * <p>贪心，将最高位的 n 与后面 m 交换，后者需满足 m>n 且 m 尽可能靠后
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/maximum-swap/solution/2021316-zui-da-jiao-huan-quan-chang-zui-ery0x/
   *
   * @param num
   * @return
   */
  public int maximumSwap(int num) {
    char[] chs = Integer.toString(num).toCharArray();
    // 记录每个数字出现的最后一次出现的下标
    int[] lastIdx = new int[10];
    for (int i = 0; i < chs.length; i++) {
      lastIdx[chs[i] - '0'] = i;
    }
    // 顺序遍历找到当前位置右边的最大的数字，并交换
    for (int i = 0; i < chs.length; i++) {
      for (int d = 9; d > chs[i] - '0'; d--) {
        if (lastIdx[d] <= i) continue;
        swap(chs, i, lastIdx[d]);
        return Integer.parseInt(chs.toString());
      }
    }
    return num;
  }

  /**
   * 最大数，把数组排成最大的数，排序 & 贪心
   *
   * <p>对 nums 按照 ab>ba 为 b>a，前导零
   *
   * <p>1.先单独证明两个数需要满足该定律，比如 3 & 30 有 330>303 显然 3 需要安排至 30 前，即表现为 3<30
   *
   * <p>2.然后证明传递性，即两两之间都要满足该性质，参考
   * https://leetcode-cn.com/problems/largest-number/solution/gong-shui-san-xie-noxiang-xin-ke-xue-xi-vn86e/
   *
   * <p>扩展1，最小数 / 把数组排成最小的数，调整本题的排序规则为 ab>ba 为 a>b 即可，参考
   * https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/
   *
   * @param nums
   * @return
   */
  public String largestNumber(int[] nums) {
    StringBuilder res = new StringBuilder();
    List<String> strs = new ArrayList<>(nums.length);
    for (int num : nums) {
      strs.add(String.valueOf(num));
    }
    strs.sort((s1, s2) -> (s2 + s1).compareTo(s1 + s2));
    for (String str : strs) {
      res.append(str);
    }
    int begin = 0;
    while (begin < nums.length - 1 && res.charAt(begin) == '0') {
      begin += 1;
    }
    return res.substring(begin);
  }

  /**
   * 移掉k位数字，字典序，单调栈，原地栈
   *
   * <p>特判移完
   *
   * <p>入栈
   *
   * @param num the num
   * @param k the k
   * @return string string
   */
  public String removeKdigits(String num, int k) {
    if (num.length() == k) {
      return "0";
    }
    StringBuilder stack = new StringBuilder();
    for (int i = 0; i < num.length(); i++) {
      char ch = num.charAt(i);
      while (k > 0 && stack.length() != 0 && stack.charAt(stack.length() - 1) > ch) {
        stack.deleteCharAt(stack.length() - 1);
        k -= 1;
      }
      if (ch == '0' && stack.length() == 0) {
        continue;
      }
      stack.append(ch);
    }
    // 判断是否移足 k 位
    String res = stack.substring(0, Math.max(stack.length() - k, 0));
    return res.length() == 0 ? "0" : res;
  }

  /**
   * 字典序的第k小数字，找到 [1,n] 内，前序
   *
   * @param n the n
   * @param k the k
   * @return int int
   */
  public int findKthNumber(int n, int k) {
    // 字典序最小即起点为 1，其前缀为 1
    int count = 1, prefix = 1;
    while (count < k) {
      // 当前 prefix 峰的值
      int curCount = count(n, prefix);
      // 本层，往下层遍历，一直遍历到第 K 个推出循环
      if (curCount + count > k) {
        prefix *= 10;
        count += 1;
      } else {
        // 去下个前缀，即相邻子树遍历
        prefix += 1;
        count += curCount;
      }
    }
    // 退出循环时 cur==k 正好找到
    return prefix;
  }

  // 统计 prefix 为前缀的层，一直向字典序更大的，遍历至 n 的个数
  // 如果说刚刚 prefix 是 1，next 是 2，那么现在分别变成 10 & 20
  // 1 为前缀的子节点增加 10 个，十叉树增加一层, 变成了两层
  // 如果说现在 prefix 是 10，next 是 20，那么现在分别变成 100 & 200
  // 1 为前缀的子节点增加 100 个，十叉树又增加了一层，变成三层
  private int count(int n, int prefix) {
    // 下一个前缀峰头，而且不断向下层遍历乘 10 可能会溢出, 所以用 long
    long cur = prefix, next = cur + 1;
    int count = 0;
    while (cur <= n) {
      // 下一峰头减去此峰头
      count += Math.min(n + 1, next) - cur;
      cur *= 10;
      // 步进至下层
      next *= 10;
    }
    return count;
  }

  /**
   * 去除重复字母 / 不同字符的最小子序列，且要求之后的整体字典序最小
   *
   * <p>greedy - 结果中第一个字母的字典序靠前的优先级最高
   *
   * <p>单调栈 - 每次贪心要找到一个当前最靠前的字母
   *
   * @param s the s
   * @return string string
   */
  public String removeDuplicateLetters(String s) {
    if (s.length() < 2) return s;
    Deque<Character> stack = new ArrayDeque<>(s.length());
    // 栈内尚存的字母
    boolean[] visited = new boolean[26];
    // 记录每个字符出现的最后一个位置
    int[] lastIdxs = new int[26];
    for (int i = 0; i < s.length(); i++) {
      lastIdxs[s.charAt(i) - 'a'] = i;
    }
    for (int i = 0; i < s.length(); i++) {
      char cur = s.charAt(i);
      if (visited[cur - 'a']) continue;
      while (!stack.isEmpty() && cur < stack.getLast() && lastIdxs[stack.getLast() - 'a'] > i) {
        visited[stack.removeLast() - 'a'] = false;
      }
      stack.addLast(cur);
      visited[cur - 'a'] = true;
    }
    StringBuilder res = new StringBuilder();
    for (char ch : stack) {
      res.append(ch);
    }
    return res.toString();
  }

  /**
   * 字典序排数
   *
   * <p>TODO
   *
   * @param n the n
   * @return list list
   */
  //  public List<Integer> lexicalOrder(int n) { }

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
   * 返回 nums[idx]，越界则 MIN_VALUE
   *
   * @param nums the nums
   * @param idx the idx
   * @return int int
   */
  protected final int curNum(int[] nums, int idx) {
    return idx <= -1 || idx >= nums.length ? Integer.MIN_VALUE : nums[idx];
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
