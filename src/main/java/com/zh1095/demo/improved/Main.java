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

  ListNode reverseList(ListNode head) {
    ListNode pre = null, cur = head;
    while (cur != null) {
      ListNode nxt = cur.next;
      cur.next = pre;
      pre = cur;
      cur = nxt;
    }
    return pre;
  }

  ListNode reverseKGroup(ListNode head, int k) {
    ListNode dummy;
    dummy.next = head;
    ListNode pre = dummy, cur = dummy;
    while (cur.next != null) {
      for (int i = 0; i < k && cur != null; i++) cur = cur.next;
      if (cur == null) break;
      ListNode curStart = pre.next, nxtStart = cur.next;
      cur.next = null;
      pre.next = reverseList(curStart);
      curStart = nxtStart;
      pre = cur = curStart;
    }
    return dummy.next;
  }
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
      for (int j = 0; j < i; j++) {
        if (nums[j] >= nums[i]) continue;
        if (dp[i] == dp[j] + 1) cnts[i] += cnts[j];
        if (dp[i] < dp[j] + 1) {
          dp[i] = dp[j] + 1;
          cnts[i] = cnts[j];
        }
      }
      maxLen = Math.max(maxLen, dp[i]);
    }
  }

  int longestPalindromeSubseq(String s) {
    for (int i = len - 1; i > -1; i--) {
      dp[i][i] = 1;
      for (int j = i + 1; j < len; j++) {
        dp[i][j] = s[i] == s[j] ? dp[i + 1][j - 1] + 2 : Math.max(dp[i + 1][j], dp[i][j - 1]);
      }
    }
    return dp[0][len - 1];
  }

  int findLength(int[] nums1, int[] nums2) {
    for (int p1 = 1; p1 <= l1; p1++) {
      for (int p2 = l2; p2 >= 1; p2--) {
        dp[p2] = nums1[p1 - 1] == nums2[p2 - 1] ? dp[p2 - 1] + 1 : 0;
        maxLen = Math.max(maxLen, dp[p2]);
      }
    }
    return maxLen;
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
    return dp[len];
  }

  int minimumTotal(int[][] triangle) {
    for (int i = len - 1; i >= 0; i--) {
      for (int j = 0; j <= i; j++) dp[j] = triangle[i][j] + Math.min(dp[j], dp[j + 1]);
    }
  }

  int integerBreak(int n) {
    for (int i = 2; i <= n; i++) {
      for (int j = 1; j <= i - 1; j++) {
        dp[i] = Math.max(dp[i], j * (i - j), j * dp[i - j]);
      }
    }
  }

  int numSquares(int n) {
    for (int i = 1; i <= n; i++) {
      dp[i] = i;
      for (int j = 1; j * j <= i; j++) {
        dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
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

  /*****************************divider*******************************/

  int longestCommonSubsequence(String text1, String text2) {
    for (int p1 = 1; p1 <= l1; p1++) {
      for (int p2 = 1; p2 <= l2; p2++) {
        dp[p1][p2] =
            text1[p1] == text2[p2]
                ? dp[p1 - 1][p2 - 1] + 1
                : Math.max(dp[p1 - 1][p2], dp[p1][p2 - 1]);
      }
    }
    return dp[l1][l2];
  }

  int editDistance(String word1, String word2) {
    for (int i = 0; i <= l1; i++) dp[i][0] = i;
    for (int j = 0; j <= l2; j++) dp[0][j] = j;
    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {
        dp[i][j] =
            word1[i - 1] == word2[j - 1]
                ? dp[i - 1][j - 1]
                : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);
      }
    }
    return dp[l1][l2];
  }

  int regularMatching(String s, String p) {
    for (int i = 1; i <= l1; i++) {
      for (int j = 1; j <= l2; j++) {}
    }
  }

  int minPathSum(int[][] grid) {
    for (int i = 1; i < grid.length; i++) {
      dp[0] += grid[i][0];
      for (int j = 1; j < COL; j++) dp[j] = Math.min(dp[j - 1], dp[j]);
    }
    return dp[COL - 1];
  }

  int uniquePaths(int m, int n) {
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        dp[j] += dp[j - 1];
      }
    }
  }

  int uniquePathsWithObstacles(int[][] obstacleGrid) {
    for (int i = 0; i < obstacleGrid.length; i++) {
      for (int j = 0; j < len; j++) {
        if (obstacleGrid[i][j] == 1) dp[j] = 0;
        if (obstacleGrid[i][j] == 0 && j > 0) dp[j] += dp[j - 1];
      }
    }
  }

  int maximalSquare(char[][] matrix) {
    for (int r = 0; r < ROW; r++) {
      int topLeft = 0;
      for (int c = 0; c < COL; c++) {
        int nxt = dp[c + 1];
        if (matrix[r][c] == '1') {
          dp[c + 1] = 1 + Math.min(topLeft, dp[c], dp[c + 1]);
          maxSide = Math.max(maxSide, dp[c + 1]);
        } else {
          dp[c + 1] = 0;
        }
        topLeft = nxt;
      }
    }
    return maxSide * maxSide;
  }

  int backToOrigin(int v, int s) {
    for (int step = 1; step < s + 1; step++) {
      for (int idx = 0; idx < v; idx++) {
        int nxt = (idx + 1) % v, tail = (idx - 1 + v) % v;
        dp[step][idx] = dp[step - 1][nxt] + dp[step - 1][tail];
      }
    }
  }

  /*****************************others*******************************/

  int longestValidParentheses(String s) {
    for (int i = 1; i < chs.length; i++) {
      if (chs[i] == '(') continue;
      int preIdx = i - dp[i - 1];
      if (preIdx > 0 && chs[preIdx - 1] == '(')
        dp[i] = 2 + dp[i - 1] + (preIdx < 2 ? 0 : dp[preIdx - 2]);
      else if (chs[i - 1] == '(') dp[i] = 2 + (i < 2 ? 0 : dp[i - 2]);
    }
    return maxLen;
  }

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
    for (int i = 1; i < s.length(); i++) {}
  }

  int candy(int[] ratings) {
    for (int i = 0; i < len; i++) l[i] = i > 0 && ratings[i] > ratings[i - 1] ? l[i - 1] + 1 : 1;
    for (int i = len - 1; i > -1; i--) {
      r = i < len - 1 && ratings[i] > ratings[i + 1] ? r + 1 : 1;
      minCnt += Math.max(l[i], r);
    }
    return minCnt;
  }
}

class TTree {
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

class AArray {
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
}

class SString {
  boolean isValid(String s) {
    for (char ch : s.toCharArray()) {
      if (!pairs.containsKey(ch)) {
        stack.offerLast(ch);
        continue;
      }
      if (stack.isEmpty() || stack.peekLast != pairs.get(ch)) return false;
      stack.pollLast();
    }
    return stack.isEmpty();
  }

  boolean checkValidString(String s) {
    for (char ch : s.toCharArray()) {
      if (ch == '*') {
        minCnt -= 1;
        maxCnt += 1;
      }
      if (minCnt < 0) minCnt = 0;
      if (minCnt > maxCnt) return false;
    }
    return minCnt == 0;
  }

  String decodeString(String s) {
    for (char ch : s.toCharArray()) {
      if (ch == '[') {
        cntStack.offerLast(cnt);
        strStack.offerLast(str.toString());
        cnt = 0;
        str = new StringBuilder();
      } else if (ch == ']') {
        int preCnt = cntStack.pollLast();
        String preStr = strStack.pollLast();
        str = new StringBuilder(preStr + str.toString().repeat(preCnt));
      } else if (ch >= '0' && ch <= '9') {
        cnt = cnt * 10 + ch - '0';
      } else {
        str.append(ch);
      }
    }
    return str.toString();
  }

  String longestCommonPrefix(String[] strs) {
    String ref = strs[0];
    for (int i = 0; i < ref.length(); i++) {
      char pivot = ref[i];
      for (int j = 1; j < strs.length; j++) {
        if (i < strs[j].length() && strs[j][i] == pivot) continue;
        return ref.substring(0, i);
      }
    }
    return ref;
  }

  String longestPalindrome(String s) {
    int start, end;
    for (int i = 0; i < 2 * len - 1; i++) {
      int lo = i / 2, hi = lo + i % 2;
      while (lo > -1 && hi < len && chs[lo] == chs[hi]) {
        if (hi - lo > end - start) {}
        lo -= 1;
        hi += 1;
      }
    }
  }

  int countSubstrings(String s) {
    int cnt;
    for (int i = 0; i < 2 * len - 1; i++) {
      int lo = i / 2, hi = lo + i % 2;
      while (lo > -1 && hi < len && chs[lo] == chs[hi]) {
        cnt += 1;
        lo -= 1;
        hi += 1;
      }
    }
  }

  int longestSubstring(String s, int k) {
    int[] counter = new int[26];
    for (char ch : s.toCharArray()) counter[ch - 'a'] += 1;
    for (char ch : s.toCharArray()) {
      if (counter[ch - 'a'] >= k) continue;
      for (String seg : s.split(String.valueOf(ch))) maxLen = max(maxLen, longestSubstring(seg, k));
      return maxLen;
    }
    return s.length();
  }

  int myAtoi(String s) {
    boolean isNegative = false;
    int idx = frontNoBlank(chs, 0);
    int n = 0;
    for (int i = idx; i < len; i++) {
      char ch = chs[i];
      if (ch < '0' || ch > '9') break;
      int pre = n;
      n = n * 10 + ch - '0';
      if (pre != n / 10) return n;
    }
    return n;
  }

  int compress(char[] chars) {
    int write, read;
    while (read < len) {
      while (start < len && chars[start] == chars[read]) start += 1;
      chars[write++] = chars[read];
      if (start - read > 1) {
        char[] cnt = Integer.toString(start - read).toCharArray();
      }
      read = start;
    }
    return write;
  }

  String reverseWords(String s) {
    reverseChs(chs, 0, len - 1);
    int start;
    while (start < len) {
      int end = frontNoBlank(chs, start);
      while (end < len && chs[end] != ' ') end += 1;
      reverseChs(chs, start, end - 1);
      start = end;
    }
    int write, read = frontNoBlank(chs, 0);
    while (read < len) {
      while (read < len && chs[read] != ' ') chs[write++] = chs[read++];
      read = frontNoBlank(chs, read);
      if (read == len) break;
      chs[write++] = ' ';
    }
    return String.valueOf(chs, 0, write);
  }

  String intToRoman(int num) {
    int cur = num;
    for (int i = 0; i < NUMs.length; i++) {
      int n = NUMs[i];
      String ch = ROMANs[i];
      while (cur >= n) {
        roman.append(ch);
        cur -= n;
      }
    }
    return roman.toString();
  }

  int[] dailyTemperatures(int[] temperatures) {
    for (int i = 0; i < temperatures.length; i++) {
      while (!ms.isEmpty() && tem[i] > tem[ms.peekLast()]) {
        int preIdx = ms.pollLast();
        tpts[preIdx] = i - preIdx;
      }
      ms.offer(i);
    }
    return new int[] {};
  }

  int[] nextGreaterElements(int[] nums) {
    for (int i = 0; i < 2 * len; i++) {
      int n = nums[i % len];
      while (!ms.isEmpty() && n > nums[ms.peekLast()]) elms[ms.pollLast()] = n;
      ms.offerLast(i % len);
    }
    return new int[] {};
  }
}
