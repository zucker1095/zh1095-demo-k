package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集数组相关，包括如下类型，部分查找不到则至 DDP
 *
 * <p>排序
 *
 * <p>寻找 & 统计
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
 * @author cenghui
 */
public class AArray extends DefaultAArray {
  /**
   * 二分查找，下方 FFind 统一该写法
   *
   * @param nums the nums
   * @param target the target
   * @return int int
   */
  public int search(int[] nums, int target) {
    // 避免当 target 小于 nums[0] & nums[end] 时多次循环运算
    if (target < nums[0] || target > nums[nums.length - 1]) return -1;
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) { // 目标可能位于碰撞点
      int mid = lo + (hi - lo) >> 1;
      if (nums[mid] < target) lo = mid + 1;
      else if (nums[mid] == target) return mid;
      else if (nums[mid] > target) hi = mid - 1;
    }
    return -1;
  }

  /**
   * 合并两个有序数组，题设不需要滤重，逆向，参考合并两个有序链表
   *
   * <p>假如需要滤重，则取代为 nextIdx
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
   * 数组中的第k个最大元素，实现小顶堆，尽量不使用内置
   *
   * <p>循环里面判断小顶堆里面的 size() 是否大于 k 个数，是的话就 poll() 出去，循环结束之后剩下来的就是 k 个数的小顶堆
   *
   * @param nums the nums
   * @param k the k
   * @return the int
   */
  public int findKthLargest(int[] nums, int k) {
    //    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    //    for (int num : nums) {
    //      minHeap.add(num);
    //      if (minHeap.size() > k) minHeap.poll();
    //    }
    //    return minHeap.peek();
    buildHeap(nums, k); // 前K个元素原地建小顶堆
    // 遍历剩下元素，比堆顶小，跳过；比堆顶大，交换后重新堆化
    for (int i = k; i < nums.length; i++) {
      if (nums[i] < nums[0]) continue;
      swap(nums, i, 0);
      heapify(nums, k, 0);
    }
    // K个元素的小顶堆的堆顶即是第K大元素
    return nums[0];
  }

  // 建堆函数 从倒数第一个非叶子节点开始堆化，其下标为 K/2-1
  private void buildHeap(int[] nums, int k) {
    for (int i = (k >> 1) - 1; i >= 0; i--) heapify(nums, k, i);
  }

  /**
   * 堆化函数 父节点下标i，左右子节点的下标分别为 2*i+1 和 2*i+2
   *
   * <p>即 down
   */
  private void heapify(int[] nums, int k, int i) {
    int cur = i, minPos = i; // 临时变量 minPos 用于存储最小值的下标，先假设父节点最小
    while (true) {
      // 和左子节点比较
      if (i * 2 + 1 < k && less(nums, i * 2 + 1, minPos)) minPos = i * 2 + 1;
      // 和右子节点比较
      if (i * 2 + 2 < k && less(nums, i * 2 + 2, minPos)) minPos = i * 2 + 2;
      // 如果 minPos 没有发生变化，说明父节点已经是最小，直接跳出
      if (minPos == i) break;
      // 否则交换
      swap(nums, i, minPos);
      // 父节点下标进行更新，继续堆化
      i = minPos;
    }
  }

  private boolean less(int[] nums, int i, int j) {
    return nums[i] < nums[j];
  }

  /**
   * 数组中的逆序对，参考归并排序，只 diff 两行代码
   *
   * @param nums the nums
   * @return int int
   */
  public int reversePairs(int[] nums) {
    int[] tmp = new int[nums.length];
    divide2(nums, tmp, 0, nums.length - 1);
    return res;
  }

  /** The Res. */
  private int res = 0;

  private void divide2(int[] nums, int[] tmp, int lo, int hi) {
    if (lo == hi) return;
    int mid = lo + (hi - lo) >> 1;
    divide2(nums, tmp, lo, mid);
    divide2(nums, tmp, mid + 1, hi);
    if (nums[mid] <= nums[mid + 1]) return;
    // diff 1
    res += mergeAndCount(nums, tmp, lo, mid, hi);
  }

  private int mergeAndCount(int[] nums, int[] tmp, int lo, int mid, int hi) {
    int res = 0;
    if (hi + 1 - lo >= 0) System.arraycopy(nums, lo, tmp, lo, hi - lo + 1);
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
        res += mid - p1 + 1;
      }
    }
    return res;
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
    int res = nums[0] + nums[1] + nums[2];
    for (int i = 0; i < nums.length; i++) {
      int lo = i + 1, hi = nums.length - 1;
      while (lo < hi) {
        int sum = nums[lo] + nums[hi] + nums[i];
        if (Math.abs(target - sum) < Math.abs(target - res)) res = sum;
        if (sum < target) lo += 1;
        else if (sum == target) return res;
        else hi -= 1;
      }
    }
    return res;
  }
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
   * 旋转数组，反转三次，全部 & [0,k-1] & [k,end]
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
}

/** 排序类 */
class SSort extends DefaultAArray {
  /**
   * 数组排序，掌握快速 & 归并 & 堆即可
   *
   * @param nums the nums
   */
  public void sortArray(int[] nums) {}
  /**
   * 快速排序，循环不变量
   *
   * @param nums the nums
   * @param lo the lo
   * @param hi the hi
   */
  public void quickSort(int[] nums, int lo, int hi) {
    if (lo >= hi) return;
    int pivotIdx = lo + new Random().nextInt(hi - lo + 1); // [lo, hi]
    swap(nums, pivotIdx, lo); // 不能省略，因为下方需要确保虚拟头也满足 < pivot
    int pivot = nums[lo];
    int lt = lo, gt = hi + 1; // 虚拟头尾，保证界外
    int cur = lt + 1;
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
    swap(nums, lo, lt); // 扰动，保证等概率分布
    quickSort(nums, lo, lt - 1);
    quickSort(nums, gt, hi);
  }

  /**
   * 归并排序，up-to-bottom 递归
   *
   * <p>bottom-to-up 迭代，参考排序链表
   *
   * @param nums the nums
   * @param lo the lo
   * @param hi the hi
   */
  public void mergeSort(int[] nums, int lo, int hi) {
    divide1(nums, new int[nums.length], 0, nums.length - 1);
  }

  private void divide1(int[] nums, int[] tmp, int lo, int hi) {
    // 对于双指针，假如迭代内部基于比较，则不需要=，假如需要统计每个元素，则需要
    if (lo >= hi) return;
    int mid = lo + (hi - lo) >> 1;
    divide1(nums, tmp, lo, mid);
    divide1(nums, tmp, mid + 1, hi);
    if (nums[mid] <= nums[mid + 1]) return;
    // curing 因为此时 [lo,mid]&[mid+1,hi] 分别有序，否则说明二者在数轴上范围存在重叠
    merge1(nums, tmp, lo, mid, hi); // 区间两两相邻合并
  }
  // 写成 < 会丢失稳定性，因为相同元素原来靠前的排序以后依然靠前，因此排序稳定性的保证必需 <=
  private void merge1(int[] nums, int[] tmp, int lo, int mid, int hi) {
    if (hi + 1 - lo >= 0) System.arraycopy(nums, lo, tmp, lo, hi + 1 - lo);
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
   * 堆排序，以下代码参考 container/heap in Go
   *
   * <p>堆是具有以下性质的完全二叉树，每个结点的值都大于或等于其左右孩子结点的值，称为大顶堆，反之，小顶堆
   */
  public class HeapSort {
    /**
     * Instantiates a new Heap sort.
     *
     * @param nums the nums
     */
    public HeapSort(int[] nums) {
      int len = nums.length;
      // 将原数组整理成堆
      heapify(nums);
      // 最大的放在末尾依次向前排序
      // 循环不变量：区间 [0, i] 堆有序
      for (int i = len - 1; i >= 1; ) {
        // 把堆顶元素（当前最大）交换到数组末尾
        // 把目前最大的数放到数组末尾, 最大的沉下去了，肯定是升序数组
        swap(nums, 0, i);
        // 逐步减少堆有序的部分
        i -= 1;
        // 下标 0 位置下沉操作，使得区间 [0, i] 堆有序
        // 从头部开始，到已经放过的数的前一个，恢复大顶堆，再次找到最大的那个数
        down(nums, 0, i);
      }
    }

    /**
     * 将数组整理成堆（堆有序）
     *
     * @param nums
     */
    private void heapify(int[] nums) {
      // 只需要从 i = (len - 1) / 2 这个位置开始逐层下移
      // 从第一个非叶子节点开始，让倒数第二层开始依次往前保证自己是个大顶堆
      // 最终到了根节点后，就全是大顶堆了
      // 此时的大顶堆，不能保证左右孩子有序
      // 数学角度考虑，完全二叉树放到数组后，各层之间的数学关系
      for (int i = (nums.length - 1) >> 1; i >= 0; i--) down(nums, i, nums.length - 1);
    }

    /**
     * @param nums
     * @param k 当前下沉元素的下标
     * @param end [0, end] 是 nums 的有效部分
     */
    private void down(int[] nums, int k, int end) {
      // n表示数组最右位的下标，防越界
      // 如果 j > n 说明当前节点是倒数第一层，所以退出循环。
      while (2 * k + 1 <= end) {
        int j = 2 * k + 1;
        // 保证后面比较且交换的是更大的那个子节点。
        // 并且从这段代码可以猜测，大顶堆从 0 开始建的话，任意节点的子节点都是 2x 和 2x+1
        if (j + 1 <= end && nums[j + 1] > nums[j]) j += 1;
        // 大于 大的那个子节点，就可以调出循环，结束down()，不用沉了
        if (nums[j] <= nums[k]) break;
        // 如果比子节点中 大的 那个 小，则交换大的到当前位置
        swap(nums, j, k);
        // 更新标记游标，再次循环计算是否需要和子节点发生交换
        k = j;
      }
    }
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
  /**
   * 打乱数组，Shuffle 即可
   *
   * @author cenghui
   */
  public class Solution {
    /** The Nums. */
    final int[] nums;

    /** The Random. */
    final Random random = new Random();

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
      for (int i = 0; i < nums.length; i++) swap(res, i, i + random.nextInt(nums.length - i));
      return res;
    }
  }
}

/**
 * 参考上方 AArray.search 的写法，即
 *
 * <p>lo<=hi
 *
 * <p>明确碰撞的含义
 */
class FFind extends DefaultAArray {
  /**
   * 寻找两个有序数组的中位数
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return double double
   */
  public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int len = nums1.length + nums2.length;
    if (len % 2 == 1) {
      int mid = len / 2;
      return getKthElement(nums1, nums2, mid + 1);
    } else {
      int mid = len / 2 - 1;
      return (getKthElement(nums1, nums2, mid + 1) + getKthElement(nums1, nums2, mid + 2)) / 2.0;
    }
  }

  private int getKthElement(int[] nums1, int[] nums2, int k) {
    int l1 = nums1.length, l2 = nums2.length;
    int p1 = 0, p2 = 0, cur = k;
    while (true) {
      // 边界情况
      if (p1 == l1) return nums2[p2 + cur - 1];
      if (p2 == l2) return nums1[p1 + cur - 1];
      if (cur == 1) return Math.min(nums1[p1], nums2[p2]);
      int half = cur / 2;
      int newIdx1 = Math.min(p1 + half, l1) - 1, newIdx2 = Math.min(p2 + half, l2) - 1;
      int num1 = nums1[newIdx1], num2 = nums2[newIdx2];
      if (num1 <= num2) {
        cur -= (newIdx1 - p1 + 1);
        p1 = newIdx1 + 1;
      } else {
        cur -= (newIdx2 - p2 + 1);
        p2 = newIdx2 + 1;
      }
    }
  }
  /**
   * 在排序数组中查找元素的第一个和最后一个位置
   *
   * @param nums the nums
   * @param target the target
   * @return int [ ]
   */
  public int[] searchRange(int[] nums, int target) {
    int[] res = new int[] {-1, -1};
    if (nums.length < 1 || nums[0] > target || nums[nums.length - 1] < target) return res;
    int lower = lowerBound(nums, target);
    if (lower == -1) return res;
    int upper = upperBound(nums, target, lower);
    return new int[] {lower, upper};
  }

  private int lowerBound(int[] nums, int target) {
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) {
      int mid = lo + ((hi - lo) >> 1);
      if (nums[mid] < target) lo = mid + 1;
      else if (nums[mid] >= target) hi = mid - 1;
    }
    return (lo > nums.length - 1 || nums[lo] != target) ? -1 : lo;
  }

  private int upperBound(int[] nums, int target, int start) {
    int lo = start, hi = nums.length - 1;
    while (lo <= hi) {
      int mid = lo + ((hi - lo) >> 1);
      if (nums[mid] <= target) lo = mid + 1;
      else if (nums[mid] > target) hi = mid - 1;
    }
    return hi;
  }

  /**
   * 搜索旋转排序数组
   *
   * <p>目标分别与中点 & 左右边界的值对比，有序的一边的边界值可能等于目标值
   *
   * @param nums the nums
   * @param target the target
   * @return int int
   */
  public int search(int[] nums, int target) {
    if (nums.length < 1) return -1;
    int lo = 0, hi = nums.length - 1;
    while (lo <= hi) {
      int mid = (lo + hi) >> 1;
      if (target == nums[mid]) return mid;
      if (nums[lo] <= nums[mid]) { // 中点的值与右边界对比，右边有序
        if (target >= nums[lo] && target < nums[mid]) hi = mid - 1; // 目标在左
        else lo = mid + 1; // 目标在右
      } else {
        if (target > nums[mid] && target <= nums[hi]) lo = mid + 1; // 目标在右
        else hi = mid - 1; // 目标在左
      }
    }
    return -1;
  }

  /**
   * 寻找旋转排序数组中的最小值，比较边界
   *
   * @param nums the nums
   * @return int int
   */
  public int findMin(int[] nums) {
    return 1;
  }

  /**
   * 寻找峰值，比较相邻
   *
   * @param nums the nums
   * @return int int
   */
  public int findPeakElement(int[] nums) {
    return 1;
  }

  /**
   * 寻找重复数，抽屉原理
   *
   * <p>据抽屉原理，小于等于 4 的个数如果严格大于 4 个，此时重复元素一定出现在 [1..4] 区间里
   *
   * @param nums the nums
   * @return int int
   */
  public int findDuplicate(int[] nums) {
    // 非索引而是值，因此相当于索引的 0
    int lo = 1, hi = nums.length - 1;
    while (lo < hi) {
      int mid = lo + ((hi - lo) >> 1);
      int count = 0;
      // 统计更小的个数
      for (int num : nums) {
        count += num > mid ? 0 : 1;
      }
      if (count > mid) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  /**
   * 缺失的第一个正数
   *
   * <p>原地哈希，nums[nums[i]-1] != nums[i]，类似数组中重复的数据
   *
   * @param nums the nums
   * @return int int
   */
  public int firstMissingPositive(int[] nums) {
    for (int i = 0; i < nums.length; i++) {
      // 不断判断 i 位置上被放入正确的数，即 nums[i]-1
      while (nums[i] > 0 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i])
        swap(nums, nums[i] - 1, i);
    }
    for (int i = 0; i < nums.length; i++) if (nums[i] != i + 1) return i + 1;
    return nums.length + 1;
  }

  /**
   * 数组中重复的数据，题设每个数字至多出现两次，且在 [1,n] 内
   *
   * <p>原地哈希，nums[nums[i]-1] *= -1，类似缺失的第一个整数
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
   * 搜索二维矩阵，模拟 BST，I & II 通用
   *
   * @param matrix the matrix
   * @param target the target
   * @return boolean boolean
   */
  public boolean searchMatrix(int[][] matrix, int target) {
    int i = 0, j = matrix[0].length - 1; // 矩阵右上角
    while (i < matrix.length && j >= 0) {
      if (matrix[i][j] < target) i += 1;
      else if (matrix[i][j] == target) return true;
      else if (matrix[i][j] > target) j -= 1;
    }
    return false;
  }
}

/** 移除 */
class DDelete extends DefaultAArray {
  /**
   * 移动零，遇到目标则跳过
   *
   * <p>需要移除的目标为 0
   *
   * <p>扩展，移除字符串中指定字符，同模板
   *
   * @param nums the nums
   */
  public void moveZeroes(int[] nums) {
    final int target = 0; // diff 1
    int last = 0, k = 0;
    for (int hi = 0; hi < nums.length; hi++) {
      if (last >= k && target == nums[hi]) continue;
      swap(nums, last, hi); // diff 2
      last += 1;
    }
  }

  private String moveChar(String s, char target) {
    char[] chs = s.toCharArray();
    int last = 0, k = 0;
    for (int hi = 0; hi < chs.length; hi++) {
      if (last >= k && target == chs[hi]) continue;
      char tmp = chs[hi]; // swap
      chs[hi] = chs[last];
      chs[last] = tmp;
      last += 1;
    }
    return Arrays.toString(Arrays.copyOfRange(chs, 0, last));
  }
  /**
   * 删除排序数组中的重复项，保留 k 位
   *
   * <p>遇到目标则跳过
   *
   * <p>扩展，参考删除字符串中的所有相邻重复项
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
   * 删除字符串中的所有相邻重复项，不保留，且需要反复执行
   *
   * <p>类似有效的括号，即括号匹配，通过 top 指针模拟栈顶，即原地栈，且修改源数组
   *
   * <p>匹配指当前 char 与栈顶不同，即入栈，否则出栈，且 skip 当前 char
   *
   * <p>最终栈内即为最终结果
   *
   * @param s the s
   * @return string string
   */
  public String removeDuplicates(String s) {
    char[] chs = s.toCharArray();
    int stackTop = -1;
    for (int i = 0; i < s.length(); i++) {
      if (stackTop == -1 || chs[stackTop] != chs[i]) {
        // 入栈
        stackTop += 1;
        chs[stackTop] = chs[i];
      } else {
        // 出栈
        stackTop -= 1;
      }
    }
    return String.valueOf(chs, 0, stackTop + 1);
  }
}

/** 字典序相关 */
class DDicOrder extends DefaultAArray {
  /**
   * 下一个排列，按照字典序
   *
   * <p>找 & 排 & 找 & 换 & 排
   *
   * @param nums the nums
   */
  public void nextPermutation(int[] nums) {
    for (int i = nums.length - 1; i > 0; i--) {
      if (nums[i] <= nums[i - 1]) continue;
      Arrays.sort(nums, i, nums.length); // 左闭右开
      for (int j = i; j < nums.length; j++) {
        if (nums[j] <= nums[i - 1]) continue;
        swap(nums, i - 1, j);
        return;
      }
    }
    Arrays.sort(nums);
  }

  /**
   * 把数组排成最小的数，对 nums 按照 s1+s2>s2+s1 排序
   *
   * <p>1.先单独证明两个数需要满足该定律，比如 3 & 30 有 303<330 显然 30 需要安排至 3 前，即 30<3
   *
   * <p>2.然后证明传递性，即两两之间都要满足该性质
   * https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/
   *
   * @param nums the nums
   * @return string string
   */
  public String minNumber(int[] nums) {
    List<String> strs = new ArrayList<>(nums.length);
    for (int num : nums) {
      strs.add(String.valueOf(num));
    }
    strs.sort((s1, s2) -> (s1 + s2).compareTo(s2 + s1));
    StringBuilder res = new StringBuilder();
    for (String str : strs) {
      res.append(str);
    }
    return res.toString();
  }

  /**
   * 移掉k位数字，字典序，单调栈
   *
   * @param num the num
   * @param k the k
   * @return string string
   */
  public String removeKdigits(String num, int k) {
    Deque<Character> stack = new ArrayDeque<>(num.length());
    for (char c : num.toCharArray()) {
      while (k > 0 && !stack.isEmpty() && c < stack.getLast()) {
        stack.removeLast();
        k -= 1;
      }
      if (c != '0' || !stack.isEmpty()) stack.addLast(c);
    }
    while (k > 0 && !stack.isEmpty()) {
      stack.removeLast();
      k -= 1;
    }
    StringBuilder res = new StringBuilder();
    for (char c : stack) res.append(c);
    return res.length() == 0 ? "0" : res.toString();
  }

  /**
   * 下一个更大元素II，单调栈 & 循环数组
   *
   * @param nums the nums
   * @return int [ ]
   */
  //  public int[] nextGreaterElements(int[] nums) {
  //    int len = nums.length;
  //    int[] res = new int[len];
  //    Arrays.fill(res, -1);
  //    Deque<Integer> stack = new ArrayDeque<>();
  //    for (int i = 0; i < 2 * len; i++) {
  //      while (!stack.isEmpty() && nums[i % len] > nums[stack.getLast()])
  //        res[stack.removeLast()] = nums[i % len];
  //      stack.addLast(i % len);
  //    }
  //    return res;
  //  }
  /**
   * 字典序的第k小数字，前序
   *
   * @param n the n
   * @param k the k
   * @return int int
   */
  public int findKthNumber(int n, int k) {
    int cur = 1; // 第一字典序小的，即 1
    int prefix = 1; // 前缀从 1 开始
    while (cur < k) {
      int tmp = count(n, prefix); // 当前prefix峰的值
      if (tmp + cur > k) { // 找到
        prefix *= 10; // 往下层遍历
        cur += 1; // 一直遍历到第K个推出循环
      } else {
        prefix += 1; // 去下个前缀遍历
        cur += tmp; // 跨过一个前缀
      }
    } // 退出循环时 cur==k 正好找到
    return prefix;
  }

  private int count(int n, int prefix) {
    // 不断向下层遍历可能一个乘 10 就溢出, 所以用 long
    long cur = prefix;
    long next = cur + 1; // 下一个前缀峰头
    int count = 0;
    while (cur <= n) {
      count += Math.min(n + 1, next) - cur; // 下一峰头减去此峰头
      // 如果说刚刚 prefix 是 1，next 是 2，那么现在分别变成 10 & 20
      // 1 为前缀的子节点增加 10 个，十叉树增加一层, 变成了两层
      // 如果说现在 prefix 是 10，next 是 20，那么现在分别变成 100 & 200
      // 1 为前缀的子节点增加 100 个，十叉树又增加了一层，变成三层
      cur *= 10;
      next *= 10; // 往下层走
    }
    return count;
  }

  /**
   * 字典序排数
   *
   * @param n the n
   * @return list list
   */
  public List<Integer> lexicalOrder(int n) {
    return new ArrayList<>();
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
    int[] lastIdxs = new int[26]; // 记录每个字符出现的最后一个位置
    boolean[] visited = new boolean[26]; // 栈内尚存的字母
    for (int i = 0; i < s.length(); i++) lastIdxs[s.charAt(i) - 'a'] = i;
    for (int i = 0; i < s.length(); i++) {
      char cur = s.charAt(i);
      if (visited[cur - 'a']) continue;
      while (!stack.isEmpty() && cur < stack.getLast() && lastIdxs[stack.getLast() - 'a'] > i)
        visited[stack.removeLast() - 'a'] = false;
      stack.addLast(cur);
      visited[cur - 'a'] = true;
    }
    StringBuilder res = new StringBuilder();
    for (char c : stack) res.append(c);
    return res.toString();
  }
  /**
   * 拼接最大数
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @param k the k
   * @return int [ ]
   */
  public int[] maxNumber(int[] nums1, int[] nums2, int k) {
    return new int[0];
  }
}

/** 提供一些数组的通用方法 */
abstract class DefaultAArray {
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
    for (int i = start + 1; i < nums.length && i > -1; i += step)
      if (nums[i] != nums[start]) return i;
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
