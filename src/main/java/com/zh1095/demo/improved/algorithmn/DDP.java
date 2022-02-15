package com.zh1095.demo.improved.algorithmn;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 收集 DP 相关
 *
 * <p>以下均为右闭期间
 *
 * <p>状态压缩尽量用具体含义，如 buy & sell 而非 dp1 & dp2
 *
 * <p>区分「以 nums[i] 结尾」&「在 [0,i-1] 区间」的 dp 定义差异
 *
 * @author cenghui
 */
public class DDP {
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
   * 分发糖果，求满足权重规则的最少所需糖果量
   *
   * <p>扩展1，成环
   *
   * @param ratings the ratings
   * @return the int
   */
  public int candy(int[] ratings) {
    int res = 0;
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
      res += Math.max(left[i], right);
    }
    return res;
  }

  /**
   * 接雨水
   *
   * @param height the height
   * @return int int
   */
  public int trap(int[] height) {
    int res = 0;
    int lm = height[0], rm = height[height.length - 1];
    for (int lo = 0, hi = height.length - 1; lo <= hi; ) {
      int left = height[lo], right = height[hi];
      lm = Math.max(lm, left);
      rm = Math.max(rm, right);
      if (left <= right) {
        res += lm - left;
        lo += 1;
      } else {
        res += rm - right;
        hi -= 1;
      }
    }
    return res;
  }

  /**
   * 盛最多水的容器
   *
   * @param height the height
   * @return int int
   */
  public int maxArea(int[] height) {
    int res = 0;
    for (int lo = 0, hi = height.length - 1; lo < hi; ) {
      int left = height[lo], right = height[hi];
      if (left <= right) {
        res = Math.max(res, left * (hi - lo));
        lo += 1;
      } else {
        res = Math.max(res, right * (hi - lo));
        hi -= 1;
      }
    }
    return res;
  }
}

/** 最优解，往状态压缩 & 双指针考量 */
class OOptimalSolution {
  /**
   * 买卖股票的最佳时机 I~III
   *
   * <p>https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/solution/zui-jian-dan-2-ge-bian-liang-jie-jue-suo-71fe/
   *
   * @param prices the prices
   * @return int int
   */
  public int maxProfit(int[] prices) {
    return maxProfitI(prices);
  }

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

  private int maxProfitII(int[] prices) {
    int buy = Integer.MIN_VALUE, sell = 0;
    for (int price : prices) {
      // 因为需要多次买卖，所以每天都要尝试是否能获得更多利润
      buy = Math.max(buy, sell - price);
      sell = Math.max(sell, buy + price);
    }
    return sell;
  }

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
   * 最小路径和，题设自然数
   *
   * <p>dp[i][j] 表示直到走到 (i,j) 的最小路径和
   *
   * <p>每次只依赖左侧和上侧的状态，因此可以压缩一维，由于不会回头，因此可以原地建立 dp
   *
   * <p>扩展1，记录路径，则需要自顶向下
   *
   * @param grid the grid
   * @return int
   */
  public int minPathSum(int[][] grid) {
    int len = grid[0].length;
    int[] dp = new int[len];
    dp[0] = grid[0][0];
    for (int i = 1; i < len; i++) {
      dp[i] = dp[i - 1] + grid[0][i];
    }
    for (int i = 1; i < grid.length; i++) {
      dp[0] = dp[0] + grid[i][0];
      for (int j = 1; j < len; j++) {
        dp[j] = Math.min(dp[j - 1] + grid[i][j], dp[j] + grid[i][j]);
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
   * @param nums the nums
   * @return int int
   */
  public int rob(int[] nums) {
    return rob2(nums);
  }

  private int rob1(int[] nums) {
    int pre = 0, cur = 0;
    for (int i : nums) {
      int tmp = Math.max(cur, pre + i);
      pre = cur;
      cur = tmp;
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
    int[] res = postOrder(root);
    return Math.max(res[0], res[1]);
  }

  private int[] postOrder(TreeNode root) {
    if (root == null) return new int[2];
    int[] left = postOrder(root.left), right = postOrder(root.right);
    return new int[] {
      Math.max(left[0], left[1]) + Math.max(right[0], right[1]), left[0] + right[0] + root.val
    };
  }

  /**
   * 最大正方形，找到只包含 1 的最大正方形
   *
   * <p>dp[i][j] 表示以 matrix[i-1][j-1] 为右下角的最大正方形的边长
   *
   * <p>递推 dp[i+1][j+1]=min(dp[i+1][j], dp[i][j+1], dp[i][j])+1)
   *
   * @param matrix the matrix
   * @return int
   */
  public int maximalSquare(char[][] matrix) {
    // 特判
    if (matrix == null || matrix.length < 1 || matrix[0].length < 1) {
      return 0;
    }
    int maxSide = 0;
    // 相当于已经预处理新增第一行、第一列均为0
    int[] dp = new int[matrix[0].length + 1];
    int northwest = 0;
    for (char[] chs : matrix) {
      northwest = 0; // 遍历每行时，还原回辅助的原值0
      for (int col = 0; col < matrix[0].length; col++) {
        int nextNorthwest = dp[col + 1];
        if (chs[col] == '1') {
          dp[col + 1] = Math.min(Math.min(dp[col], dp[col + 1]), northwest) + 1;
          // maxSide = max(maxSide, dp[row+1][col+1]);
          maxSide = Math.max(maxSide, dp[col + 1]);
        } else {
          dp[col + 1] = 0;
        }
        northwest = nextNorthwest;
      }
    }
    return maxSide * maxSide;
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
   * <p>扩展1，不能爬到 7 倍数的楼层
   *
   * <p>扩展2，记录爬楼梯的路径，选用 dfs
   *
   * @param n the n
   * @return int int
   */
  public int climbStairs(int n) {
    int step1 = 1, step2 = 1; // dp[i-1] & dp[i-2]
    for (int i = 2; i < n + 1; i++) {
      // 扩展1，无法状态压缩
      // if ((i + 1) % 7 == 0) { dp[i] = 0 }
      // else { dp[i] = dp[i - 1] + dp[i - 2] }
      int tmp = step2;
      step2 = step2 + step1;
      step1 = tmp;
    }
    return step2;
  }

  private int dfs(int n, int[] memo) {
    if (n == 0 || n == 1) {
      return 1;
    }
    return memo[n] == 0 ? dfs(n - 1, memo) + dfs(n - 2, memo) : memo[n];
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
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i < n + 1; i++) for (int j = 1; j < i + 1; j++) dp[i] += dp[j - 1] * dp[i - j];
    return dp[n];
  }

  /**
   * 解码方法
   *
   * <p>TODO
   *
   * <p>dp[i] 表示 str[0,i] 的解码总数
   *
   * <p>递推关系
   *
   * @param s the s
   * @return int
   */
  public int numDecodings(String s) {
    // 补充前导
    s = " " + s;
    char[] chs = s.toCharArray();
    int[] dp = new int[s.length() + 1];
    dp[0] = 1;
    for (int i = 1; i <= s.length(); i++) {
      // a 代表「当前位置」单独形成 item
      // b 代表「当前位置」与「前一位置」共同形成 item
      int a = chs[i] - '0', b = (chs[i - 1] - '0') * 10 + (chs[i] - '0');
      // 如果 a 属于有效值，那么 dp[i] 可以由 dp[i - 1] 转移过来
      if (1 <= a && a <= 9) {
        dp[i] = dp[i - 1];
      }
      // 如果 b 属于有效值，则 dp[i] 可以由 dp[i - 2] 或者 dp[i - 1] & dp[i - 2] 转移过来
      if (10 <= b && b <= 26) {
        dp[i] += dp[i - 2];
      }
    }
    return dp[s.length()];
  }
}

/**
 * 路径相关，其余如海岛类 & 最长递增路径，参考 TTree
 *
 * @author cenghui
 */
class PPath {
  /**
   * 不同路径I
   *
   * <p>dp[i][j] 表示由起点，即 [0][0] 达到 [i][j] 的路径总数
   *
   * @param m the m
   * @param n the n
   * @return int int
   */
  public int uniquePaths(int m, int n) {
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    for (int i = 1; i < m; i++) for (int j = 1; j < n; j++) dp[j] += dp[j - 1];
    return dp[n - 1];
  }

  /**
   * 不同路径II
   *
   * <p>dp[i][j] 表示由起点，即 [0][0] 达到 [i][j] 的路径总数
   *
   * @param obstacleGrid the obstacle grid
   * @return int int
   */
  public int uniquePathsWithObstacles(int[][] obstacleGrid) {
    int m = obstacleGrid[0].length;
    int[] dp = new int[m];
    dp[0] = (obstacleGrid[0][0] == 1) ? 0 : 1; // 起点可能有障碍物
    for (int[] ints : obstacleGrid) {
      for (int j = 0; j < m; ++j) {
        if (ints[j] == 1) dp[j] = 0;
        else if (ints[j] == 0 && j - 1 >= 0) dp[j] = dp[j] + dp[j - 1];
      }
    }
    return dp[m - 1];
  }
}

/**
 * 子数组，连续，子序列，不连续，即子数组相当于连续子序列
 *
 * <p>最长上升子序列(LIS):Longest Increasing Subsequence
 *
 * <p>最长连续序列(LCTS):Longest Consecutive Sequence
 *
 * <p>最长连续递增序列(LCIS):Longest Continuous Increasing Subsequence
 *
 * <p>最长公共子序列(LCMS):Longest Common Subsequence
 *
 * <p>前缀和参考 https://leetcode-cn.com/circle/discuss/SrePlc/
 *
 * @author cenghui
 */
class SSubArray {
  /**
   * 最大子数组和 / 最大子序和 / 连续子数组的最大和，前缀和
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
    // int lo = 0, hi = 0;
    //      while (hi < nums.length-1) {
    //      if (curSum > 0) {
    //        curSum += num;
    //      } else {
    //        curSum = num;
    //        lo = hi;
    //      }
    //      hi += 1;
    // }
    return res;
  }

  /**
   * 和为k的子数组，前缀和 value：key 对应的前缀和的个数
   *
   * @param nums the nums
   * @param k the k
   * @return int int
   */
  public int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> freqByPreSum = new HashMap() {};
    // 对于下标为 0 的元素，前缀和为 0，个数为 1
    freqByPreSum.put(0, 1);
    int preSum = 0, count = 0;
    for (int num : nums) {
      preSum += num;
      // 先获得前缀和为 preSum - k 的个数，加到计数变量里
      if (freqByPreSum.containsKey(preSum - k)) count += freqByPreSum.get(preSum - k);
      // 然后维护 preSumFreq 的定义
      freqByPreSum.put(preSum, freqByPreSum.getOrDefault(preSum, 0) + 1);
    }
    return count;
  }

  /**
   * 最长有效括号，需要考虑上一个成对的括号区间
   *
   * <p>三类题型，生成回溯，判断栈，找最长 dp
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
   * 最长重复子数组，最长公共子串同一代码，区分楼上子序列的代码
   *
   * <p>dp[i][j] 表示 A[0:i-1] & B[0:j-1] 的最长公共前缀
   *
   * <p>递推 dp[i][j] = (nums1[i - 1] == nums2[j - 1]) ? dp[i - 1][j - 1] + 1 : 0;
   *
   * <p>状态压缩，由于是连续，因此递推关系只依赖前一个变量，类似滑窗
   *
   * @param nums1 the nums 1
   * @param nums2 the nums 2
   * @return int int
   */
  public int findLength(int[] nums1, int[] nums2) {
    int res = 0;
    // int[][] dp = new int[nums1.length + 1][nums2.length + 1];
    int[] dp = new int[nums2.length + 1];
    for (int i = 1; i <= nums1.length; i++) {
      for (int j = nums2.length; j >= 1; j--) {
        dp[j] = nums1[i - 1] == nums2[j - 1] ? dp[j - 1] + 1 : 0;
        res = Math.max(res, dp[j]);
      }
      // for (int j = 1; j <= nums2.length; j++) {
      //   dp[i][j] = (nums1[i - 1] == nums2[j - 1]) ? dp[i - 1][j - 1] + 1 : 0;
      //   res = Math.max(res, dp[i][j]);
      //  }
    }
    return res;
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
   * @return int
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

/** The type S sub sequence. */
class SSubSequence {
  /**
   * 最长递增子序列 / 最长上升子序列
   *
   * <p>[0,m-1] & [0,j-1]
   *
   * <p>dp[i] 表示 nums[:i-1] 的最长递增子序列
   *
   * <p>扩展1，时间复杂度为 nlogn 参考
   * https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/dong-tai-gui-hua-er-fen-cha-zhao-tan-xin-suan-fa-p/
   *
   * <p>扩展2，输出，分别记录 i & j 即可，参下 annotate
   *
   * @param nums the nums
   * @return int int
   */
  public int lengthOfLIS(int[] nums) {
    int res = 0;
    if (nums.length == 0) {
      return res;
    }
    int[] dp = new int[nums.length];
    Arrays.fill(dp, 1);
    for (int i = 0; i < nums.length; i++) {
      int pivot = nums[i];
      //      int idx = 0;
      for (int j = 0; j < i; j++) {
        if (nums[j] < pivot) {
          //          idx = j;
          dp[i] = Math.max(dp[i], dp[j] + 1);
        }
      }
      //      if (dp[i] >= res) {
      //        // 记录 i & j
      //      }
      res = Math.max(res, dp[i]);
    }
    return res;
  }

  /**
   * 最长连续序列，并查集 & 哈希
   *
   * <p>map[i] 表示以 i 为端点的最长连续序列
   *
   * @param nums the nums
   * @return int int
   */
  public int longestConsecutive(int[] nums) {
    Map<Integer, Integer> lenAsVertex = new HashMap<>(nums.length);
    int res = 0;
    for (int num : nums) {
      int left = lenAsVertex.get(num - 1), right = lenAsVertex.get(num + 1);
      if (lenAsVertex.containsKey(num)) continue;
      int cur = 1 + left + right;
      res = Math.max(res, cur);
      lenAsVertex.put(num, cur);
      lenAsVertex.put(num - left, cur);
      lenAsVertex.put(num + right, cur);
    }
    return res;
  }

  /**
   * 最长公共子序列
   *
   * <p>与楼下模板一致，[1,m] & [1,n]
   *
   * <p>dp[i][j] 表示 A[0:i-1] & B[0:j-1] 的最长公共前缀
   *
   * <p>扩展1，求最长公共子串的长度，即最长连续序列，参上
   *
   * <p>扩展2，输出该子序列，则补充首尾指针，参下 annotate
   *
   * @param text1 the text 1
   * @param text2 the text 2
   * @return int int
   */
  public int longestCommonSubsequence(String text1, String text2) {
    int n1 = text1.length(), n2 = text2.length();
    int[][] dp = new int[n1 + 1][n2 + 1];
    //    int lo = 0, hi = 0;
    for (int i = 1; i <= n1; i++) {
      for (int j = 1; j <= n2; j++) {
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
    return dp[n1][n2];
  }

  /**
   * 编辑距离 & 两个字符串的删除操作，均是 LCS 最长公共子序列的问题
   *
   * <p>与楼上模板一致，[1,m] & [1,n]
   *
   * @param word1 the word 1
   * @param word2 the word 2
   * @return int int
   */
  public int minDistance(String word1, String word2) {
    return editDistance(word1, word2);
  }

  // 编辑距离
  // dp[i][j] 表示由 A[0:i] 转移为 B[0:j] 的最少步数
  // 递推 min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1
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
            (word1.charAt(i - 1) == word2.charAt(j - 1))
                ? dp[i - 1][j - 1]
                : Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
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
   * 判断子序列
   *
   * <p>扩展1，依次检查海量 s 是否均为 t 的子序列
   *
   * <p>KMP 思想，类似于用伪链表把相同的字符给链接起来，如对于 abac
   *
   * <p>1、算法实现过程如下： 1.1 填充字符 " " => ' abac' 1.2 对其中的字符a链表而言（a-z每个字符都执行一次下述操作,共26次） dp[3]['a'-'a'] =>
   * dp[3][0] = -1 记录a最近的一次位置为，nexPos = 3 dp[1]['a'-'a'] => dp[1][0] = 3 记录a最近的一次位置为，nexPos = 1
   * dp[0][0] = 1 (预处理填充的空字符意义所在，否则初始位置的a就找不到了)
   *
   * <p>2、查找子串过程（） 2.1 初始索引为0,遍历待查找子串 2.2 查找 aa 的过程如下 idx = 0 （从idx+1以及之后的位置开始查找） idx = dp[0][c-'a']
   * => idx = dp[0][0] => idx = 1 idx = dp[idx][c-'a'] => dp[1][0] = 3 此时 aa 已经遍历完，返回true 上述过程，只要idx
   * = -1,表示找不到字符，则返回false
   *
   * @param s pattern
   * @param t main
   * @return boolean
   */
  public boolean isSubsequence(String s, String t) {
    // 预处理以保证 t[0] 也被正确表示，即 dp[0][..]
    t = " " + t;
    int[][] dp = new int[t.length()][26];
    for (int ch = 0; ch < 26; ch++) { // 每一轮处理一个字符
      int nxtPos = -1;
      for (int i = t.length() - 1; i > -1; i--) {
        dp[i][ch] = nxtPos;
        nxtPos = (t.charAt(i) == ch + 'a') ? nxtPos : i;
      }
    }
    // 起始位置是空字符（idx = 0）
    // dp[0][p] 表示从 1 开始查找 p+'a' 在 t 中的位置
    int idx = 0;
    for (char ch : s.toCharArray()) {
      idx = dp[idx][ch - 'a'];
      if (idx == -1) {
        return false;
      }
    }
    return true;
  }

  /**
   * 正则表达式匹配，以下均基于 p 判定
   *
   * <p>TODO
   *
   * <p>dp[i][j] 表示 s[0,i-1] 能否被 p[0,j-1] 匹配
   *
   * <p>dp[i][j] = dp[i-1][j] 多个字符匹配的情况
   *
   * <p>dp[i][j] = dp[i][j-1] 单个字符匹配的情况
   *
   * <p>dp[i][j] = dp[i][j-2] 没有匹配的情况
   *
   * @param s the s
   * @param p the p
   * @return boolean
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
