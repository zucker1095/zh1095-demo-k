package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集 DP 相关
 *
 * <p>以下均为右闭期间
 *
 * <p>状态压缩基于滚动数组，尽量用具体含义，如 buy & sell 而非 dp1 & dp2
 *
 * <p>区分「以 nums[i] 结尾」&「在 [0,i-1] 区间」的 dp 定义差异
 *
 * @author cenghui
 */
public class DDP {}

/**
 * 子数组，连续，子序列，不连续，即子数组相当于连续子序列
 *
 * <p>连续子区间可考虑前缀和，常搭配哈希表，因为希望遍历得到前缀和的时候，一边遍历一边记住结果，参考 https://leetcode-cn.com/circle/discuss/SrePlc/
 *
 * <p>最长上升子序列(LIS):Longest Increasing Subsequence
 *
 * <p>最长连续序列(LCTS):Longest Consecutive Sequence
 *
 * <p>最长连续递增序列(LCIS):Longest Continuous Increasing Subsequence
 *
 * <p>最长公共子序列(LCMS):Longest Common Subsequence
 *
 * @author cenghui
 */
class OptimalSubArray {
  /**
   * 最长重复子数组 / 最长公共子串，区别于子序列的代码
   *
   * <p>dp[i][j] 表示 A[0:i-1] & B[0:j-1] 的最长公共前缀
   *
   * <p>递推 dp[i][j]=(nums1[i - 1]==nums2[j - 1]) ? dp[i - 1][j - 1]+1 : 0;
   *
   * <p>状态压缩，由于是连续，因此递推关系只依赖前一个变量，类似滑窗
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return int int
   */
  public int findLength(int[] nums1, int[] nums2) {
    int maxLen = 0;
    // int[][] dp = new int[nums1.length + 1][nums2.length + 1];
    int[] dp = new int[nums2.length + 1];
    for (int i = 1; i <= nums1.length; i++) {
      for (int j = nums2.length; j >= 1; j--) {
        dp[j] = nums1[i - 1] == nums2[j - 1] ? dp[j - 1] + 1 : 0;
        maxLen = Math.max(maxLen, dp[j]);
      }
      // for (int j = 1; j <= nums2.length; j++) {
      //   dp[i][j] = (nums1[i - 1] == nums2[j - 1]) ? dp[i - 1][j - 1] + 1 : 0;
      //   res = Math.max(res, dp[i][j]);
      //  }
    }
    return maxLen;
  }

  /**
   * 目标和，找到 nums 一个正子集与一个负子集，使其总和等于 target，统计这种可能性的总数
   *
   * <p>公式推出，找到一个正数集 P，其和的两倍，等于目标和 + 序列总和，即 01 背包
   *
   * <p>dp[j] 表示填满 j 容积的包的方案数，即组合
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/target-sum/solution/dai-ma-sui-xiang-lu-494-mu-biao-he-01bei-rte9/
   *
   * <p>扩展1，改为乘法
   *
   * <p>扩展2，target 为负
   *
   * @param nums the nums
   * @param target the target
   * @return int
   */
  public int findTargetSumWays(int[] nums, int target) {
    int sum = 0;
    for (int num : nums) {
      sum += num;
    }
    if ((target + sum) % 2 != 0) return 0;

    int maxCapacity = (target + sum) / 2;
    if (maxCapacity < 0) maxCapacity *= -1;

    int[] dp = new int[maxCapacity + 1];
    dp[0] = 1;
    for (int volume : nums) {
      for (int capacity = maxCapacity; capacity >= volume; capacity--) {
        dp[capacity] += dp[capacity - volume];
      }
    }
    return dp[maxCapacity];
  }

  /**
   * 乘积最大子数组，可能存在负数，因此至少需要引入两个状态
   *
   * <p>dp[i][0] 表示以 nums[i] 结尾的子数组的乘积的最小值，dp[i][1] 为最大
   *
   * <p>递推需要根据 nums[i] 判断
   *
   * <p>递归关系只与前一个相关，因此滚动变量，即状态压缩第一维，而保留 0 & 1 两个状态
   *
   * @param nums the nums
   * @return int int
   */
  public int maxProduct(int[] nums) {
    int res = Integer.MIN_VALUE;
    int multiMax = 1, multiMin = 1;
    for (int num : nums) {
      // 以该点结尾的乘积大小调换
      if (num < 0) {
        int tmp = multiMax;
        multiMax = multiMin;
        multiMin = tmp;
      }
      multiMax = Math.max(multiMax * num, num);
      multiMin = Math.min(multiMin * num, num);
      res = Math.max(res, multiMax);
    }
    return res;
  }
}

/** 子序列 */
class OptimalSubSequence extends DichotomyClassic {
  /**
   * 最长递增子序列 / 最长上升子序列，基于贪心
   *
   * <p>如果想让上升子序列尽量的长，那么需要每次在上升子序列末尾添加的数字尽可能小，如 3465 应该选 345 而非 346
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-dong-tai-gui-hua-2/
   * 与
   * https://leetcode-cn.com/problems/pile-box-lcci/solution/ti-mu-zong-jie-zui-chang-shang-sheng-zi-7jfd3/
   *
   * <p>扩展1，输出路径，参考
   * https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/xiao-bai-lang-jing-dian-dong-tai-gui-hua-px0v/
   *
   * @param nums the nums
   * @return int int
   */
  public int lengthOfLIS(int[] nums) {
    // tail 最后一个已经赋值的元素的索引
    int len = nums.length, end = 0;
    // tail[i] 表示长度为 i+1 的所有上升子序列的结尾的最小数字，如 3465 中 tail[1]=4
    int[] tail = new int[len], dp = new int[len];
    tail[0] = nums[0];
    // dp[0] = 1;
    for (int i = 1; i < len; i++) {
      // 插入或替换
      if (nums[i] <= tail[end]) {
        int lo = lowerBound(Arrays.copyOfRange(tail, 0, end + 1), nums[i]);
        tail[lo] = nums[i];
        // dp[i] = lo + 1;
      } else {
        end += 1;
        tail[end] = nums[i];
        // dp[i] = end + 1;
      }
    }
    // findPath(nums,dp,end+1);
    // 题目要求返回的是长度，因此 +1 后返回
    return end + 1;
  }

  // 需要反向从后往前找，因为相同长度的 dp[i]，后面的肯定比前面的字典序小
  // 如果后面的比前面大，那么必定后面的长度 > 前面的长度
  // https://www.nowcoder.com/questionTerminal/9cf027bf54714ad889d4f30ff0ae5481?answerType=1&f=discussion
  // https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/dong-tai-gui-hua-er-fen-cha-zhao-tan-xin-suan-fa-p/
  private int[] findPath(int[] nums, int[] dp, int len) {
    int[] path = new int[len];
    int count = len;
    for (int i = nums.length - 1; i >= 0 && count >= 0; i--) {
      if (dp[i] != count) continue;
      count -= 1;
      path[count] = nums[i];
    }
    return path;
  }

  /**
   * 最长连续序列
   *
   * <p>参考
   * https://leetcode-cn.com/problems/longest-consecutive-sequence/solution/xiao-bai-lang-ha-xi-ji-he-ha-xi-biao-don-j5a2/
   *
   * @param nums the nums
   * @return int int
   */
  public int longestConsecutive(int[] nums) {
    int maxLen = 0;
    // 所在连续区间的长度
    Map<Integer, Integer> lens = new HashMap<>(nums.length);
    for (int num : nums) {
      if (lens.containsKey(num)) continue;
      // 左右连续区间，与当前数字所在连续区间的长度
      int left = lens.getOrDefault(num - 1, 0),
          right = lens.getOrDefault(num + 1, 0),
          cur = 1 + left + right;
      // 表示已经遍历过该值
      lens.put(num, -1);
      // 分别更新当前连续区间左右边界对应的区间长度
      lens.put(num - left, cur);
      lens.put(num + right, cur);
      maxLen = Math.max(maxLen, cur);
    }
    return maxLen;
  }

  /**
   * 最长公共子序列，[1,m] & [1,n]
   *
   * <p>dp[i][j] 表示 A[0:i-1] & B[0:j-1] 的最长公共前缀
   *
   * <p>扩展1，求最长公共子串的长度，参上「最长连续序列」
   *
   * <p>扩展2，输出该子序列，则补充首尾指针，参下 annotate
   *
   * @param text1 the text 1
   * @param text2 the text 2
   * @return int int
   */
  public int longestCommonSubsequence(String text1, String text2) {
    int len1 = text1.length(), len2 = text2.length();
    int[][] dp = new int[len1 + 1][len2 + 1];
    //    int lo = 0, hi = 0;
    for (int i = 1; i <= len1; i++) {
      for (int j = 1; j <= len2; j++) {
        dp[i][j] =
            text1.charAt(i - 1) == text2.charAt(j - 1)
                ? dp[i - 1][j - 1] + 1
                : Math.max(dp[i - 1][j], dp[i][j - 1]);
        //        if (hi - lo + 1 < j - i + 1) {
        //          i = lo;
        //          j = hi;
        //        }
      }
    }
    return dp[len1][len2];
  }

  /**
   * 编辑距离 & 两个字符串的删除操作，均是 LCS 最长公共子序列的问题
   *
   * <p>与「最长公共子序列」模板一致，[1,m] & [1,n]
   *
   * @param word1 the word 1
   * @param word2 the word 2
   * @return int int
   */
  public int minDistance(String word1, String word2) {
    return editDistance(word1, word2);
  }

  // 编辑距离，画图即可，三个方向分别代表三种操作
  // dp[i][j] 表示由 A[0:i] 转移为 B[0:j] 的最少步数
  // 递推 1+min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
  // 扩展1，三个操作权重不同，求最少的总权重
  private int editDistance(String word1, String word2) {
    int n1 = word1.length(), n2 = word2.length();
    int[][] dp = new int[n1 + 1][n2 + 1];
    for (int i = 0; i <= n1; i++) {
      dp[i][0] = i;
    }
    for (int j = 0; j <= n2; j++) {
      dp[0][j] = j;
    }
    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
        dp[i][j] =
            word1.charAt(i - 1) == word2.charAt(j - 1)
                ? dp[i - 1][j - 1]
                : 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
      }
    }
    return dp[n1][n2];
    //    int ic, rc, dc;
    //    int[] dp2 = new int[n2 + 1];
    //    // 初始化第一行
    //    for (int i = 1; i <= n2; i++) {
    //      dp2[i] = i * ic;
    //    }
    //    for (int i = 1; i <= n1; i++) {
    //      int pre = dp2[0];
    //      dp2[0] = i * dc;
    //      for (int j = 1; j <= n2; ++j) {
    //        int tmp = dp2[j]; // 上一轮 dp[i-1][j]
    //        if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
    //          dp2[j] = pre;
    //        } else {
    //          dp2[j] = Math.min(pre + rc, Math.min(dp2[j - 1] + ic, tmp + dc));
    //        }
    //        pre = tmp; // 更新 dp[i-1][j-1]
    //      }
    //    }
  }

  // 两个字符串的删除操作
  // dp[i][j] 表示由 s1[0:i] 转移为 s2[0:j] 的最少步数
  // 递推 min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1])+1
  // min(dp
  private int deleteOperationForTwoStrings(String s1, String s2) {
    int n1 = s1.length(), n2 = s2.length();
    int[][] dp = new int[n1 + 1][n2 + 1];
    for (int i = 0; i <= n1; i++) {
      dp[i][0] = i;
    }
    for (int j = 0; j <= n2; j++) {
      dp[0][j] = j;
    }
    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
        int minDistance = Math.min(dp[i - 1][j], dp[i][j - 1]) + 1;
        dp[i][j] =
            (s1.charAt(i - 1) == s2.charAt(j - 1))
                ? Math.min(minDistance, dp[i - 1][j - 1])
                : minDistance;
      }
    }
    return dp[n1][n2];
  }

  /**
   * 正则表达式匹配，以下均基于 p 判定，类似「通配符匹配」
   *
   * <p>TODO dp[i][j] 表示 s[0,i-1] 能否被 p[0,j-1] 匹配
   *
   * <p>dp[i-1][j] 多个字符匹配的情况，dp[i][j-1] 单个字符匹配的情况，dp[i][j-2] 没有匹配的情况
   *
   * @param s the s
   * @param p the p
   * @return boolean boolean
   */
  public boolean isMatch(String s, String p) {
    boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
    dp[0][0] = true;
    for (int i = 0; i < p.length(); i++) {
      dp[0][i + 1] = (p.charAt(i) == '*' && dp[0][i - 1]);
    }
    for (int i = 0; i < s.length(); i++) {
      for (int j = 0; j < p.length(); j++) {
        // 如果是任意元素 or 是对于元素匹配
        if (p.charAt(j) == s.charAt(i) || p.charAt(j) == '.') {
          dp[i + 1][j + 1] = dp[i][j];
        }
        // 如果前一个元素不匹配且不为任意元素
        if (p.charAt(j) == '*') {
          dp[i + 1][j + 1] =
              (p.charAt(j - 1) == s.charAt(i) || p.charAt(j - 1) == '.')
                  ? dp[i + 1][j] || dp[i][j + 1] || dp[i + 1][j - 1]
                  : dp[i + 1][j - 1];
        }
      }
    }
    return dp[s.length()][p.length()];
  }
}

/**
 * 路径相关，其余如海岛类 & 最长递增路径，参考 TTree
 *
 * @author cenghui
 */
class OptimalPath {
  /**
   * 最小路径和，题设自然数
   *
   * <p>参考
   * https://leetcode-cn.com/problems/minimum-path-sum/solution/dong-tai-gui-hua-lu-jing-wen-ti-ni-bu-ne-fkil/0/
   *
   * <p>dp[i][j] 表示 (0,0) to (i,j) 的最小路径和
   *
   * <p>每次只依赖左侧和上侧的状态，因此可以压缩一维，由于不会回头，因此可以原地建立 dp
   *
   * <p>扩展1，记录路径，则需要自底向上，即从右下角，终点开始遍历，参下 annotate
   *
   * <p>扩展2，存在负值点
   *
   * @param grid the grid
   * @return int int
   */
  public int minPathSum(int[][] grid) {
    int len = grid[0].length;
    int[] dp = new int[len];
    dp[0] = grid[0][0];
    for (int i = 1; i < len; i++) {
      dp[i] = dp[i - 1] + grid[0][i];
    }
    for (int i = 1; i < grid.length; i++) {
      dp[0] += grid[i][0];
      for (int j = 1; j < len; j++) {
        dp[j] = Math.min(dp[j - 1], dp[j]) + grid[i][j];
      }
    }
    //    List<Integer> res = new ArrayList<>();
    //    int i = grid.length - 1, j = grid[0].length - 1;
    //    res.add(grid[i][j]);
    //    int sum = dp[i][j];
    //    while (i > 0 || j > 0) {
    //      sum -= grid[i][j];
    //      if (j - 1 >= 0 && dp[i][j - 1] == sum) {
    //        res.add(grid[i][j - 1]);
    //        j -=1;
    //      } else {
    //        res.add(grid[i - 1][j]);
    //        i -=1;
    //      }
    //    }
    //    return res;
    return dp[len - 1];
  }

  /**
   * 三角形的最小路径和，bottom to up
   *
   * <p>dp[i][j] 表示由 triangle[0][0] 达到 triangle[i][j] 的最小路径长度
   *
   * @param triangle the triangle
   * @return int int
   */
  public int minimumTotal(List<List<Integer>> triangle) {
    int len = triangle.get(triangle.size() - 1).size();
    int[] dp = new int[len + 1];
    for (int i = len - 1; i >= 0; i--) {
      for (int j = 0; j <= i; j++) {
        dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
      }
    }
    return dp[0];
  }

  /**
   * 打家劫舍
   *
   * <p>dp[i] 表示 nums[0,i] 产生的最大金额
   *
   * <p>dp[i] = Math.max(dp[i-1], nums[i-1] + dp[i-2]);
   *
   * <p>扩展1，记录路径，参下 annotate
   *
   * @param nums the nums
   * @return int int
   */
  public int rob(int[] nums) {
    return rob2(nums);
  }

  private int rob1(int[] nums) {
    int pre = 0, cur = 0;
    //    StringBuilder pathPre = new StringBuilder(), pathCur = new StringBuilder();
    for (int num : nums) {
      //      if (cur > pre + num) {
      //        pathCur = new StringBuilder(pathPre.toString());
      //        cur = cur;
      //      } else {
      //        String tmp1 = pathPre.toString();
      //        pathPre = new StringBuilder(pathCur.toString()).append(num);
      //        pathCur = new StringBuilder(tmp1);
      //        int tmp2 = Math.max(cur, pre + num);
      //        pre = cur;
      //        cur = tmp2;
      //      }
      int tmp2 = Math.max(cur, pre + num);
      pre = cur;
      cur = tmp2;
    }
    return cur;
  }

  private int rob2(int[] nums) {
    if (nums.length < 2) return nums[0];
    return Math.max(
        rob1(Arrays.copyOfRange(nums, 0, nums.length - 1)),
        rob1(Arrays.copyOfRange(nums, 1, nums.length)));
  }

  /**
   * 打家劫舍III，树状，后序遍历，两种选择，遍历与否当前点
   *
   * @param root the root
   * @return int int
   */
  public int rob(TreeNode root) {
    int[] res = dfs11(root);
    return Math.max(res[0], res[1]);
  }

  private int[] dfs11(TreeNode root) {
    if (root == null) return new int[2];
    int[] left = dfs11(root.left), right = dfs11(root.right);
    int cur = left[0] + right[0] + root.val;
    int nxt = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
    return new int[] {nxt, cur};
  }
}

/** 最优解，状态压缩 & 双指针 */
class OptimalElse {
  /**
   * 买卖股票的最佳时机 I~III
   *
   * <p>参考
   * https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/solution/zui-jian-dan-2-ge-bian-liang-jie-jue-suo-71fe/
   *
   * @param prices the prices
   * @return int int
   */
  public int maxProfit(int[] prices) {
    return maxProfitI(prices);
  }

  // 限定一次
  private int maxProfitI(int[] prices) {
    int buy = Integer.MIN_VALUE, sell = 0;
    for (int price : prices) {
      // max(不买，买了)
      buy = Math.max(buy, -price);
      // max(不卖，卖了)
      sell = Math.max(sell, buy + price);
    }
    return sell;
  }

  // 不限次数
  private int maxProfitII(int[] prices) {
    int buy = Integer.MIN_VALUE, sell = 0;
    for (int price : prices) {
      // 因为需要多次买卖，所以每天都要尝试是否能获得更多利润
      buy = Math.max(buy, sell - price);
      sell = Math.max(sell, buy + price);
    }
    return sell;
  }

  // 限定两次，综合上方二者
  private int maxProfitIII(int[] prices) {
    // 因为只能交易 2 次，所以定义 2 组 buy & sell
    int buy1 = Integer.MIN_VALUE, sell1 = 0;
    int buy2 = Integer.MIN_VALUE, sell2 = 0;
    for (int price : prices) {
      // 第一次买 -p
      buy1 = Math.max(buy1, -price);
      // 第一次卖 buy1+p
      sell1 = Math.max(sell1, buy1 + price);
      // 第一次卖了后现在买 sell1-p
      buy2 = Math.max(buy2, sell1 - price);
      // 第二次买了后现在卖 buy2+p
      sell2 = Math.max(sell2, buy2 + price);
    }
    return sell2;
  }

  /**
   * 接雨水，贪心，漏桶效应
   *
   * @param height the height
   * @return int int
   */
  public int trap(int[] height) {
    int lo = 0, hi = height.length - 1;
    int volume = 0, lm = height[lo], rm = height[hi];
    while (lo < hi) {
      int left = height[lo], right = height[hi];
      lm = Math.max(lm, left);
      rm = Math.max(rm, right);
      if (left <= right) {
        volume += lm - left;
        lo += 1;
      } else {
        volume += rm - right;
        hi -= 1;
      }
    }
    return volume;
  }

  /**
   * 盛最多水的容器
   *
   * @param height the height
   * @return int int
   */
  public int maxArea(int[] height) {
    int maxVolume = 0;
    for (int lo = 0, hi = height.length - 1; lo < hi; ) {
      int left = height[lo], right = height[hi];
      if (left <= right) {
        maxVolume = Math.max(maxVolume, left * (hi - lo));
        lo += 1;
      } else {
        maxVolume = Math.max(maxVolume, right * (hi - lo));
        hi -= 1;
      }
    }
    return maxVolume;
  }

  /**
   * 最长有效括号，需要考虑上一个成对的括号区间
   *
   * <p>括号相关的参考「括号生成」与「有效的括号」
   *
   * <p>dp[i] 表示 s[0,i-1] 的最长有效括号
   *
   * @param s the s
   * @return int int
   */
  public int longestValidParentheses(String s) {
    int res = 0;
    int[] dp = new int[s.length()];
    for (int i = 1; i < s.length(); i++) {
      if (s.charAt(i) == '(') {
        continue;
      }
      int preCount = i - dp[i - 1];
      if (s.charAt(i - 1) == '(') {
        dp[i] = i >= 2 ? dp[i - 2] + 2 : 2;
      } else if (preCount > 0 && s.charAt(preCount - 1) == '(') {
        dp[i] = dp[i - 1] + (preCount >= 2 ? dp[preCount - 2] + 2 : 2);
      }
      res = Math.max(res, dp[i]);
    }
    return res;
  }

  /**
   * 零钱兑换，硬币可重复，与下方 II 保持外 coin 内 amount
   *
   * <p>dp[i] 表示凑成 i 元需要的最少的硬币数
   *
   * @param coins the coins
   * @param amount the amount
   * @return int int
   */
  public int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    // 因为要比较的是最小值，这个不可能的值就得赋值成为一个最大值
    Arrays.fill(dp, amount + 1);
    // 单独一枚硬币如果能够凑出面值，符合最优子结构
    dp[0] = 0;
    Arrays.sort(coins);
    for (int coin : coins) {
      for (int i = coin; i <= amount; i++) {
        if (i < coin) break;
        dp[i] = Math.min(dp[i], dp[i - coin] + 1);
      }
    }
    return (dp[amount] == amount + 1) ? -1 : dp[amount];
  }

  /**
   * 完全平方数，完全背包，类似「零钱兑换」
   *
   * <p>dp[i] 表示和为 i 的几个完全平方数的最少数量，如 13=4+9 则 dp[13] 为 2
   *
   * @param n the n
   * @return int
   */
  public int numSquares(int n) {
    int[] dp = new int[n + 1];
    for (int i = 1; i <= n; i++) {
      // 至少全由 1 组成
      dp[i] = i;
      for (int j = 1; j * j <= i; j++) {
        dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
      }
    }
    return dp[n];
  }

  /**
   * 森林中的兔子
   *
   * <p>TODO
   *
   * <p>先对所有出现过的数字进行统计，然后再对数值按颜色分配
   *
   * @param cs
   * @return
   */
  public int numRabbits(int[] answers) {
    Map<Integer, Integer> counter = new HashMap<Integer, Integer>();
    for (int answer : answers) {
      counter.put(answer, counter.getOrDefault(answer, 0) + 1);
    }
    int count = 0;
    for (Map.Entry<Integer, Integer> entry : counter.entrySet()) {
      int y = entry.getKey(), x = entry.getValue();
      count += (x + y) / (y + 1) * (y + 1);
    }
    return count;
  }

  /**
   * 分发糖果，求满足权重规则的最少所需糖果量
   *
   * <p>扩展1，成环
   *
   * @param ratings the ratings
   * @return the int
   */
  public int candy(int[] ratings) {
    int minCount = 0;
    int[] left = new int[ratings.length];
    for (int i = 0; i < ratings.length; i++) {
      //      if (i == 0 && ratings[0] > ratings[ratings.length - 1]) {
      //        left[i] = left[ratings.length - 1] + 1;
      //        continue;
      //      }
      left[i] = (i > 0 && ratings[i] > ratings[i - 1]) ? left[i - 1] + 1 : 1;
    }
    int right = 0;
    for (int i = ratings.length - 1; i >= 0; i--) {
      //      if (i == ratings.length - 1 && ratings[0] < ratings[i]) {
      //        right += 1;
      //        continue;
      //      }
      right = (i < ratings.length - 1 && ratings[i] > ratings[i + 1]) ? right + 1 : 1;
      minCount += Math.max(left[i], right);
    }
    return minCount;
  }

  /**
   * 使序列递增的最小交换次数
   *
   * <p>对于位置 i 至少满足以下两种情况之一
   *
   * <p>A[i]>A[i-1] && B[i]>B[i-1] 与 A[i]>B[i-1] && B[i]>A[i-1]
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/minimum-swaps-to-make-sequences-increasing/solution/leetcode-801-wo-gan-jio-ying-gai-jiang-de-hen-tou-/
   *
   * @param nums1
   * @param nums2
   * @return
   */
  public int minSwap(int[] nums1, int[] nums2) {
    // 分别保存不交换与交换的最小操作次数
    int keep = 0, swap = 1;
    for (int i = 1; i < nums1.length; i++) {
      if (nums1[i - 1] < nums1[i] && nums2[i - 1] < nums2[i]) {
        if (nums1[i - 1] < nums2[i] && nums2[i - 1] < nums1[i]) {
          keep = Math.min(keep, swap);
          swap = keep + 1;
        } else {
          swap += 1;
        }
      } else {
        int pre = keep;
        keep = swap;
        swap = pre + 1;
      }
    }
    return Math.min(keep, swap);
  }

  /**
   * 鸡蛋掉落
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/super-egg-drop/solution/ji-ben-dong-tai-gui-hua-jie-fa-by-labuladong/
   *
   * @param K
   * @param N
   * @return
   */
  public int superEggDrop(int K, int N) {
    // m 最多不会超过 N 次（线性扫描）
    int[][] dp = new int[K + 1][N + 1];
    // base case
    // dp[0][..] = 0
    // dp[..][0] = 0
    int celling = 0;
    while (dp[K][celling] < N) {
      celling += 1;
      for (int k = 1; k <= K; k++) {
        dp[k][celling] = dp[k][celling - 1] + dp[k - 1][celling - 1] + 1;
      }
    }
    return celling;
  }
}

/**
 * 统计
 *
 * <p>区分统计排列 & 组合的区别
 */
class CCount {
  /**
   * 爬楼梯，对比零钱兑换 II，可选集为 [1,2] 需要返回凑成 n 的总数，元素可重，前者排列，后者组合
   *
   * <p>先走 2 步再走 1 步与先 1 后 2 是两种爬楼梯的方案，而先拿 2 块再拿 1 块 & 相反是同种凑金额的方案
   *
   * <p>扩展1，不能爬到 7 倍数的楼层，参下 annotate
   *
   * <p>扩展2，记录爬楼梯的路径，参下回溯
   *
   * <p>扩展3，爬 1 or 3 级，参考
   * https://www.nowcoder.com/questionTerminal/1e6ac1a96c3149348aa9009709a36a6f?f=discussion
   *
   * @param n the n
   * @return int int
   */
  public int climbStairs(int n) {
    int step1 = 1, step2 = 1; // dp[i-1] & dp[i-2]
    for (int i = 2; i < n + 1; i++) {
      // 扩展 1 无法状态压缩
      // if ((i + 1) % 7 == 0) { dp[i] = 0 }
      // else { dp[i] = dp[i - 1] + dp[i - 2] }
      int tmp = step2;
      step2 = step2 + step1;
      step1 = tmp;
    }
    return step2;
  }

  private void backtracking14(int floor, StringBuilder path, List<String> res) {
    if (floor == 0) {
      res.add(path.toString());
      return;
    }
    for (int step = 1; step <= 2; step++) {
      int nxtFloor = floor - step;
      if (nxtFloor < 0) break;
      path.append(nxtFloor);
      backtracking14(nxtFloor, path, res);
      path.deleteCharAt(path.length() - 1);
    }
  }

  /**
   * 零钱兑换II，返回可以凑成总金额的硬币组合数，硬币可重，即 coins[i] 同个索引可重复选择
   *
   * <p>dp[i] 表示凑成 i 元的路径总数，即组合
   *
   * <p>与上方 I 保持外 coin 内 amount
   *
   * <p>https://leetcode-cn.com/problems/coin-change-2/solution/ling-qian-dui-huan-iihe-pa-lou-ti-wen-ti-dao-di-yo/
   *
   * @param amount the amount
   * @param coins the coins
   * @return int int
   */
  public int change(int amount, int[] coins) {
    int[] dp = new int[amount + 1];
    dp[0] = 1;
    for (int coin : coins) for (int i = coin; i <= amount; i++) dp[i] += dp[i - coin];
    return dp[amount];
  }

  /**
   * 不同路径I
   *
   * <p>dp[i][j] 表示由起点，即 [0,0] 达到 [i,j] 的路径总数
   *
   * @param m the m
   * @param n the n
   * @return int int
   */
  public int uniquePaths(int m, int n) {
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        dp[j] += dp[j - 1];
      }
    }
    return dp[n - 1];
  }

  /**
   * 不同路径II
   *
   * <p>dp[i][j] 表示由起点，即 (0,0) to (i,j) 的路径总数
   *
   * @param obstacleGrid the obstacle grid
   * @return int int
   */
  public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int len = obstacleGrid[0].length;
    int[] dp = new int[len];
    // 起点可能有障碍物
    dp[0] = obstacleGrid[0][0] == 1 ? 0 : 1;
    for (int[] rows : obstacleGrid) {
      for (int i = 0; i < len; i++) {
        if (rows[i] == 1) {
          dp[i] = 0;
        } else if (rows[i] == 0 && i >= 1) {
          dp[i] = dp[i] + dp[i - 1];
        }
      }
    }
    return dp[len - 1];
  }

  /**
   * 圆环回原点，类似爬楼梯，参考 https://mp.weixin.qq.com/s/NZPaFsFrTybO3K3s7p7EVg
   *
   * <p>圆环上有 m 个点，编号为 0~m-1，从 0 点出发，每次可以逆时针和顺时针走一步，求 n 步回到 0 点的走法
   *
   * <p>dp[i][j] 表示从 0 出发走 i 步到达 j 点的方案，即排列数
   *
   * <p>递推，走 n 步到 0 的方案数 = 走 n-1 步到 1 的方案数 + 走 n-1 步到 m-1 的方案数
   *
   * @param m 点数
   * @param n 步数
   * @return int int
   */
  public int backToOrigin(int m, int n) {
    // 便于从 1 开始递推
    int[][] dp = new int[m][n + 1];
    dp[0][0] = 1;
    // j+1 or j-1 可能越界 [0, m-1] 因此取余
    for (int step = 1; step < n + 1; step++) {
      for (int idx = 0; idx < m; idx++) {
        int idxNxt = (idx + 1) % m, idxTail = (idx - 1 + m) % m;
        dp[step][idx] = dp[step - 1][idxNxt] + dp[step - 1][idxTail];
      }
    }
    return dp[n][0];
  }

  /**
   * 不同的二叉搜索树，卡特兰数公式，记忆即可
   *
   * <p>dp[i] 表示假设 i 个节点存在二叉排序树的个数
   *
   * @param n the n
   * @return int int
   */
  public int numTrees(int n) {
    int[] dp = new int[n + 1];
    dp[0] = dp[1] = 1;
    for (int i = 2; i < n + 1; i++) {
      for (int j = 1; j < i + 1; j++) {
        dp[i] += dp[j - 1] * dp[i - j];
      }
    }
    return dp[n];
  }

  /**
   * 解码方法，返回字符可以被编码的方案总数，如对于 "226" 可以被解码为 "2 26" & "22 6" & "2 2 6"
   *
   * <p>参考
   * https://leetcode-cn.com/problems/decode-ways/solution/c-wo-ren-wei-hen-jian-dan-zhi-guan-de-jie-fa-by-pr/
   *
   * <p>dp[i] 表示 str[0,i] 的解码总数
   *
   * <p>递推关系，按照如下顺序，分别判断正序遍历时，当前与前一位的数字，s[i]=='0' -> s[i-1]=='1' or '2'
   *
   * <p>显然 dp[i] 仅依赖前二者，因此可状态压缩
   *
   * @param s the s
   * @return int int
   */
  public int numDecodings(String s) {
    if (s.charAt(0) == '0') return 0;
    // dp[-1]=dp[0]=1
    int pre = 1, cur = 1;
    for (int i = 1; i < s.length(); i++) {
      char curCh = s.charAt(i), preCh = s.charAt(i - 1);
      int tmp = cur;
      if (curCh == '0') {
        if (preCh != '1' && preCh != '2') {
          return 0;
        }
        cur = pre;
      } else if (preCh == '1' || (preCh == '2' && curCh >= '1' && curCh <= '6')) {
        cur += pre;
      }
      pre = tmp;
    }
    return cur;
  }
}

/** 收集矩形相关，矩阵参下路径 */
class OptimalRectangle {
  /**
   * 最大正方形，找到只包含 1 的最大正方形
   *
   * <p>dp[i][j] 表示以 matrix[i-1][j-1] 为右下角的最大正方形的边长
   *
   * <p>递推 dp[i+1][j+1]=min(dp[i+1][j], dp[i][j+1], dp[i][j])+1)
   *
   * <p>只需关注当前格子的周边，故可二维降一维优化，参考
   * https://leetcode-cn.com/problems/maximal-square/solution/li-jie-san-zhe-qu-zui-xiao-1-by-lzhlyle/
   *
   * @param matrix the matrix
   * @return int int
   */
  public int maximalSquare(char[][] matrix) {
    int maxSide = 0;
    // 相当于已经预处理新增第一行、第一列均 0
    int[] dp = new int[matrix[0].length + 1];
    for (char[] row : matrix) {
      // 左上角
      int topLeft = 0;
      for (int col = 0; col < row.length; col++) {
        int nxtTopLeft = dp[col + 1];
        if (row[col] == '1') {
          dp[col + 1] = 1 + Math.min(Math.min(dp[col], dp[col + 1]), topLeft);
          // maxSide = max(maxSide, dp[row+1][col+1]);
          maxSide = Math.max(maxSide, dp[col + 1]);
        } else {
          dp[col + 1] = 0;
        }
        topLeft = nxtTopLeft;
      }
    }
    return maxSide * maxSide;
  }

  /**
   * 柱状图中最大的矩形
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/largest-rectangle-in-histogram/solution/zhao-liang-bian-di-yi-ge-xiao-yu-ta-de-zhi-by-powc/
   *
   * @param heights
   * @return
   */
  public int largestRectangleArea(int[] heights) {
    int maxArea = 0;
    Deque<Integer> stack = new ArrayDeque<>();
    int[] newHeights = new int[heights.length + 2];
    for (int i = 1; i < heights.length + 1; i++) {
      newHeights[i] = heights[i - 1];
    }
    for (int i = 0; i < newHeights.length; i++) {
      while (!stack.isEmpty() && newHeights[stack.peekLast()] > newHeights[i]) {
        int cur = stack.pollLast(), pre = stack.peekLast();
        maxArea = Math.max(maxArea, (i - pre - 1) * newHeights[cur]);
      }
      stack.offerLast(i);
    }
    return maxArea;
  }

  /**
   * 最大矩形
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/maximal-rectangle/solution/yu-zhao-zui-da-ju-xing-na-ti-yi-yang-by-powcai/
   *
   * @param matrix
   * @return
   */
  public int maximalRectangle(char[][] matrix) {
    int maxArea = 0, row = matrix.length, col = matrix[0].length;
    int[] newHeights = new int[col + 2];
    for (int i = 0; i < row; i++) {
      Deque<Integer> stack = new ArrayDeque<>();
      for (int j = 0; j < col + 2; j++) {
        if (j >= 1 && j <= col) {
          newHeights[j] = matrix[i][j - 1] == '1' ? newHeights[j] + 1 : 0;
        }
        while (!stack.isEmpty() && newHeights[stack.peekLast()] > newHeights[i]) {
          int cur = stack.pollLast(), pre = stack.peekLast();
          maxArea = Math.max(maxArea, (j - pre - 1) * newHeights[cur]);
        }
        stack.offerLast(j);
      }
    }
    return maxArea;
  }
}
