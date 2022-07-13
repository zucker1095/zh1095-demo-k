package com.zh1095.demo.improved;

import java.util.*;

public class Main {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    String input = in.nextLine(); // [1,2,3,4]
    int[] arr = new int[input.length() - 2];
    for (String seg : input.substring(1, input.length() - 2).split(",")) {}
    System.out.println(arr);
    // 回车后
    int cnt = Integer.parseInt(in.nextLine()); // 2
    System.out.println(cnt);
  }

  void quickSort(int[] nums, int lo, int hi) {
    if (lo >= hi) return;
    int pivotIdx = lo + new Random().nextInt(hi - lo + 1);
    int pivot = nums[pivotIdx];
    swap(nums, pivotIdx, lo);
    int lt = lo, cur = lt + 1, gt = hi + 1;
    while (cur < gt) {
      if (nums[cur] < pivot) {
        lt += 1;
        swap(nums, cur, lt);
        cur += 1;
      }
      if (nums[cur] == pivot) cur += 1;
      if (nums[cur] > pivot) {
        gt -= 1;
        swap(nums, cur, lt);
      }
    }
    quickSort(nums, lo, lt - 1);
    quickSort(nums, gt, hi);
  }

  int findKthLargest(int[] nums, int k) {
    heapify(nums, k - 1);
    for (int i = k; i < nums.length; i++) {
      if (priorityThan(nums[i], nums[0])) continue;
      swap(nums, 0, i);
      sink(nums, 0, k - 1);
    }
    return nums[0];
  }

  void heapSort(int[] nums) {
    heapify(nums, nums.length - 1);
    for (int i = nums.length - 1; i > 0; i++) {
      swap(nums, 0, i);
      sink(nums, 0, i - 1);
    }
  }

  int minMeetingRooms(int[][] intervals) {
    PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    Arrays.sort(intervals, (v1, v2) -> v1[0] - v2[0]);
    minHeap.offer(intervals[0][1]);
    for (int i = 1; i < intervals.length; i++) {
      if (minHeap.peek() <= intervals[i][0]) minHeap.poll();
      minHeap.offer(intervals[i][1]);
    }
    return minHeap.size();
  }

  class MedianFinder {
    private final PriorityQueue<Integer> minHeap = new PriorityQueue<>((a, b) -> a - b),
        maxHeap = new PriorityQueue<>((a, b) -> b - a);

    public void addNum(int num) {}

    public double findMedian() {}
  }

  boolean hasPathSum(TreeNode root, int sum) {
    if (root == null) return false;
    int v = root.val;
    if (root.left == null && root.right == null) return v == sum;
    return hasPathSum(root.left, sum - v) || hasPathSum(root.right, sum - v);
  }

  List<List<Integer>> pathSum(TreeNode root, int targetSum) {
    List<List<Integer>> paths = new ArrayList<>();
    bt1(root, new ArrayDeque<>(), paths, targetSum);
    return paths;
  }

  void bt1(TreeNode root, Deque<Integer> path, List<List<Integer>> res, int target) {}
}

class DP {
  int lengthOfLIS(int[] nums) {
    int end = 0;
    int[] tails = new int[len];
    tails[0] = nums[0];
    for (int n : nums) {
      if (n > tails[end]) {
        end += 1;
        tails[end] = n;
      } else {
        int lo = lowerBound(nums, 0, end, n);
        tails[lo] = n;
      }
    }
    return end + 1;
  }

  int findNumberOfLIS(int[] nums) {
    for (int i = 0; i < len; i++) {
      dp[i] = cnts[i] = 1;
      for (int j = 0; j < i; j++) {}
    }
  }

  int longestPalindromeSubseq(String s) {
    for (int i = len - 1; i > -1; i--) {
      dp[i][i] = 1;
      for (int j = i + 1; j < len; j++) {
        if (s.charAt(i) == s.charAt(j)) {

        } else {

        }
      }
    }
  }

  int findLength(int[] nums1, int[] nums2) {
    for (int p1 = 1; p1 <= l1; p1++) {
      for (int p2 = l2; p2 >= 1; p2--) {
        if (nums1[p1] == nums2[p2]) {

        } else {

        }
        maxLen = Math.max(maxLen, dp[p2]);
      }
    }
  }

  boolean workBreak(String s, List<String> wordDict) {
    for (int i = 1; i < len + 1; i++) {
      for (int j = i; j > -1; j--) {
        if (j + maxLen < i) break;
        if (dp[j] & wordSet.contains(s.substring(j, i))) {
          dp[i] = true;
          break;
        }
      }
    }
  }

  int minimumTotal(int[][] triangle) {
    for (int i = len - 1; i >= 0; i--) {
      for (int j = 0; j <= i; j++) {
        dp[j] = triangle[i][j] + Math.min(dp[j], dp[j + 1]);
      }
    }
  }

  int numTrees(int n) {
    for (int i = 2; i < n + 1; i++) {
      for (int j = 1; j < i + 1; j++) {
        dp[i] += dp[j - 1] * dp[i - j];
      }
    }
  }

  int integerBreak(int n) {
    for(int i = 2; i <= n; i++) {
      for(int j = 1; j <= i - 1 ; j++) {

      }
    }
  }

  int numSquares(int n) {
    for (int i = 1; i <= n; i++) {
      dp[i] = i;
      for (int j = 1; j * j <= i; j++){}
    }
  }

  /*****************************divider*******************************/

  int longestCommonSubsequence(String text1, String text2) {
    for (int p1 = 0; p1 <= l1; p1++) {
      for (int p2 = 0; p2 <= l2; p2++) {
        if (text1.charAt(p1) == text2.charAt(p2)) {
          dp[p1][p2] = dp[p1-1][p2-1] +1;
        } else {
          dp[p1][p2]
        }
      }
    }
  }

  int editDistance(String word1, String word2) {
    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {
        if (word1.charAt(i) == word2.charAt(j)) {

        } else {

        }
      }
    }
  }

  int regularMatching(String s, String p) {
    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {}
    }
  }

  int minPathSum(int[][] grid) {
    for (int i = 1; i < grid.length; i++) {
      dp[0] += grid[i][0];
      for (int j = 1; j < COL; j++) {}
    }
  }

  int uniquePaths(int m, int n) {
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {}
    }
  }

  int uniquePathsWithObstacles(int[][] obstacleGrid) {
    for (int i = 0; i < obstacleGrid.length; i++) {
      for (int j = 0; j < len; j++) {}
    }
  }

  int maximalSquare(char[][] matrix) {
    for (int r = 0; r < matrix.length; r++) {
      int topLeft = 0;
      for (int c = 0; c < matrix[0].length; c++) {}
    }
  }

  int backToOrigin(int v, int s) {
    for (int step = 1; step < s + 1; step++) {
      for (int idx = 0; idx < v; idx++) {

      }
    }
  }

  /*****************************others*******************************/

  int longestConsecutive(int[] nums) {
    for (int n : nums) set.add(n);
    int maxLen = 0;
    for (int n : set) {
      if (set.contains(n - 1)) continue;
      int upper = n;
      while (set.contains(upper + 1)) upper += 1;
      maxLen = Math.max(maxLen, upper - n + 1);
    }
    return maxLen;
  }

  int coinChange(int[] coins, int amount) {
    for (int c : coins) {
      for (int i = c; i <= amount; i++) {}
    }
  }

  int numDecodings(String s) {
    for (int i = 1; i < s.length(); i++) {

    }
  }
}
