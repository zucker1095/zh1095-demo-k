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

  public String convert(String s, int numRows) {
    int r = 0, isForward = -1;
    for (char ch : s.toCharArray()) {
      rows[r].append(ch);
      if (r == 0 || r == numRows - 1) isForward *= -1;
      r += isForward;
    }
    for (StringBuilder i : rows) res.append(i);
    return res.toString();
  }

  List<Integer> partitionLabels(String s) {
    for (int i = 0; i < chs.length; i++) lastIdxes[chs[i] - 'a'] = i;
    int lo, hi;
    for (int i = 0; i < chs.length; i++) {
      hi = Math.max(hi, lastIdxes[chs[i] - 'a']);
      if (i > hi) continue;
      lens.add(hi - lo + 1);
      lo = hi + 1;
    }
    return lens;
  }

  int mySqrt(int x) {
    while (true) {
      double cur = (n + x / n) * 0.5;
      if (Math.abs(n - cur) < 1e-7) break;
      n = cur;
    }
    return (int) n;
  }

  double myPow(double x, int n) {
    return n < 0 ? 1.0 / quickMulti(x, -n) : quickMulti(x, n);
  }

  double quickMulti(double x, int n) {
    if (n == 0) return 1;
    double y = quickMulti(x, n / 2);
    return y * y * (((n & 1) == 0) ? 1 : x);
  }

  int translateNum(int num) {
    if (num <= 9) return 1;
    int ba = num % 100, res = translateNum(num / 10); // xyzcba
    return ba > 9 && ba < 26 ? res + translateNum(num / 100) : res;
  }

  int majorityElement(int[] nums) {
    for (int n : nums) {
      if (cnt == 0) {
        can = n;
        cnt = 1;
      } else if (n == cnt) {
        cnt += 1;
      } else {
        cnt -= 1;
      }
    }
    return can;
  }

  int findNthDigit(int n) {
    while (n > (long) len * 9 * base) {
      n -= len * 9 * base;
      len += 1;
      base *= 10;
    }
    int idx = n - 1, digit = idx % len;
    double num = Math.pow(10, len - 1) + idx / len;
    return (int) (num / Math.pow(10, len - digit - 1) % 10);
  }

  int lastRemaining(int n, int m) {
    int leftIdx = 0;
    for (int i = 2; i <= n; i++) leftIdx = (leftIdx + m) % i;
    return leftIdx;
  }

  String fractionToDecimal(int num, int de) {
    for (long n = num % de; n != 0; n %= de) {
      if (de2Idx.containsKey(n)) {
        res.insert(de2Idx.get(n), "(");
        res.append(")");
        break;
      }
      de2Idx.put(n, res.length());
      n *= 10;
      res.append(n / de);
    }
    return res.toString();
  }
}

//class DP {
//  int lengthOfLIS(int[] nums) {
//    int end = 0;
//    int[] tails = new int[len];
//    tails[0] = nums[0];
//    for (int n : nums) {
//      if (n > tails[end]) {
//        end += 1;
//        tails[end] = n;
//      } else {
//        int lo = lowerBound(nums, 0, end, n);
//        tails[lo] = n;
//      }
//    }
//    return end + 1;
//  }
//
//  int findNumberOfLIS(int[] nums) {
//    for (int i = 0; i < len; i++) {
//      dp[i] = cnts[i] = 1;
//      for (int j = 0; j < i; j++) {
//        if (nums[j] >= nums[i]) continue;
//        if (dp[i] == dp[j] + 1) cnts[i] += cnts[j];
//        if (dp[i] < dp[j] + 1) {
//          dp[i] = dp[j] + 1;
//          cnts[i] = cnts[j];
//        }
//      }
//      maxLen = Math.max(maxLen, dp[i]);
//    }
//    int cnt = 0;
//    for (int i = 0; i < len; i++) {
//      if (dp[i] == maxLen) cnt += cnts[i];
//    }
//  }
//
//  int longestPalindromeSubseq(String s) {
//    for (int i = len - 1; i > -1; i--) {
//      dp[i][i] = 1;
//      for (int j = i + 1; j < len; j++) {
//        dp[i][j] = s[i] == s[j] ? dp[i + 1][j - 1] + 2 : Math.max(dp[i + 1][j], dp[i][j - 1]);
//      }
//    }
//    return dp[0][len - 1];
//  }
//
//  int findLength(int[] nums1, int[] nums2) {
//    for (int p1 = 1; p1 <= l1; p1++) {
//      for (int p2 = l2; p2 >= 1; p2--) {
//        dp[p2] = nums1[p1 - 1] == nums2[p2 - 1] ? dp[p2 - 1] + 1 : 0;
//        maxLen = Math.max(maxLen, dp[p2]);
//      }
//    }
//    return maxLen;
//  }
//
//  boolean workBreak(String s, List<String> wordDict) {
//    for (int i = 1; i < len + 1; i++) {
//      for (int j = i; j > -1; j--) {
//        if (j + maxLen < i) break;
//        if (dp[j] & wordSet.contains(s.substring(j, i))) {
//          dp[i] = true;
//          break;
//        }
//      }
//    }
//    return dp[len];
//  }
//
//  int minimumTotal(int[][] triangle) {
//    for (int i = len - 1; i >= 0; i--) {
//      for (int j = 0; j <= i; j++) dp[j] = triangle[i][j] + Math.min(dp[j], dp[j + 1]);
//    }
//  }
//
//  int integerBreak(int n) {
//    for (int i = 2; i <= n; i++) {
//      for (int j = 1; j <= i - 1; j++) {
//        dp[i] = Math.max(dp[i], j * (i - j), j * dp[i - j]);
//      }
//    }
//  }
//
//  int numSquares(int n) {
//    for (int i = 1; i <= n; i++) {
//      dp[i] = i;
//      for (int j = 1; j * j <= i; j++) {
//        dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
//      }
//    }
//  }
//
//  int numTrees(int n) {
//    dp[0] = dp[1] = 1;
//    for (int i = 2; i < n + 1; i++) {
//      for (int j = 1; j < i + 1; j++) {
//        dp[i] += dp[j - 1] * dp[i - j];
//      }
//    }
//  }
//
//  /*****************************divider*******************************/
//
//  int longestCommonSubsequence(String text1, String text2) {
//    for (int p1 = 1; p1 <= l1; p1++) {
//      for (int p2 = 1; p2 <= l2; p2++) {
//        dp[p1][p2] =
//            text1[p1] == text2[p2]
//                ? dp[p1 - 1][p2 - 1] + 1
//                : Math.max(dp[p1 - 1][p2], dp[p1][p2 - 1]);
//      }
//    }
//    return dp[l1][l2];
//  }
//
//  int editDistance(String word1, String word2) {
//    for (int i = 0; i <= l1; i++) dp[i][0] = i;
//    for (int j = 0; j <= l2; j++) dp[0][j] = j;
//    for (int i = 1; i <= l1; i++) {
//      for (int j = 1; j <= l2; j++) {
//        dp[i][j] =
//            word1[i - 1] == word2[j - 1]
//                ? dp[i - 1][j - 1]
//                : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);
//      }
//    }
//    return dp[l1][l2];
//  }
//
//  int regularMatching(String s, String p) {
//    dp[0][0] = true;
//    for (int j = 1; j <= l2; j++) dp[0][j] = pChs[j - 1] == '*' && dp[0][j - 2];
//    for (int i = 1; i <= l1; i++) {
//      for (int j = 1; j <= l2; j++) {
//        char s = sChs[i - 1], p = pChs[j - 1];
//        if (p == s || p == '.') dp[i][j] = dp[i - 1][j - 1];
//        if (p == '*') {
//          char pre = pChs[j - 2];
//          if (pre == s || pre == '.') dp[i][j] = dp[i][j - 2] || dp[i - 1][j];
//          else dp[i][j] = dp[i][j - 2];
//        }
//      }
//    }
//  }
//
//  int minPathSum(int[][] grid) {
//    for (int i = 1; i < grid.length; i++) {
//      dp[0] += grid[i][0];
//      for (int j = 1; j < COL; j++) dp[j] = Math.min(dp[j - 1], dp[j]);
//    }
//    return dp[COL - 1];
//  }
//
//  int uniquePaths(int m, int n) {
//    for (int i = 1; i < m; i++) {
//      for (int j = 1; j < n; j++) {
//        dp[j] += dp[j - 1];
//      }
//    }
//  }
//
//  int uniquePathsWithObstacles(int[][] obstacleGrid) {
//    for (int i = 0; i < obstacleGrid.length; i++) {
//      for (int j = 0; j < len; j++) {
//        if (obstacleGrid[i][j] == 1) dp[j] = 0;
//        if (obstacleGrid[i][j] == 0 && j > 0) dp[j] += dp[j - 1];
//      }
//    }
//  }
//
//  int maximalSquare(char[][] matrix) {
//    for (int r = 0; r < ROW; r++) {
//      int topLeft = 0;
//      for (int c = 0; c < COL; c++) {
//        int nxt = dp[c + 1];
//        if (matrix[r][c] == '1') {
//          dp[c + 1] = 1 + Math.min(topLeft, dp[c], dp[c + 1]);
//          maxSide = Math.max(maxSide, dp[c + 1]);
//        } else dp[c + 1] = 0;
//        topLeft = nxt;
//      }
//    }
//    return maxSide * maxSide;
//  }
//
//  int backToOrigin(int v, int s) {
//    for (int step = 1; step < s + 1; step++) {
//      for (int idx = 0; idx < v; idx++) {
//        int nxt = (idx + 1) % v, tail = (idx - 1 + v) % v;
//        dp[step][idx] = dp[step - 1][nxt] + dp[step - 1][tail];
//      }
//    }
//  }
//
//  /*****************************others*******************************/
//  int maxProfit(int[] prices) {
////    buys[1] = max(buys[1], -p) & sells[1] = max(sells[1], buys[1]+p)
//    for (int p : prices) {
//      for (int i = 1; i <= k; ++i) {
//        buys[i] = Math.max(buys[i], sells[i - 1] - p); // 卖了买
//        sells[i] = Math.max(sells[i], buys[i] + p); // 买了卖
//      }
//    }
//    return sells[k];
//  }
//
//  int findLengthOfLCIS(int[] nums) {
//    int len = nums.length, maxLen = 1, curLen = 1;
//    if (len < 2) return len;
//    for (int i = 0; i < len - 1; i++) {
//      if (nums[i + 1] > nums[i]) curLen += 1;
//      else curLen = 1;
//      maxLen = Math.max(maxLen, curLen);
//    }
//    return maxLen;
//  }
//
//  int maxProduct(int[] nums) {
//    for (int n : nums) {
//      if (n < 0) {
//        int tmp = proMax;
//        proMax = proMin;
//        proMin = tmp;
//      }
//      proMax = Math.max(proMax * n, n);
//      proMin = Math.min(proMin * n, n);
//      maxPro = Math.max(maxPro, proMax);
//    }
//  }
//
//  int longestValidParentheses(String s) {
//    for (int i = 1; i < chs.length; i++) {
//      if (chs[i] == '(') continue;
//      int preIdx = i - dp[i - 1];
//      if (preIdx > 0 && chs[preIdx - 1] == '(')
//        dp[i] = 2 + dp[i - 1] + (preIdx < 2 ? 0 : dp[preIdx - 2]);
//      else if (chs[i - 1] == '(') dp[i] = 2 + (i < 2 ? 0 : dp[i - 2]);
//    }
//    return maxLen;
//  }
//
//  int longestConsecutive(int[] nums) {
//    for (int n : nums) set.add(n);
//    int maxLen = 0;
//    for (int n : set) {
//      if (set.contains(n - 1)) continue;
//      int upper = n;
//      while (set.contains(upper + 1)) upper += 1;
//      maxLen = Math.max(maxLen, upper - n + 1);
//    }
//    return maxLen;
//  }
//
//  int coinChange(int[] coins, int amount) {
//    for (int c : coins) {
//      for (int i = c; i <= amount; i++) {}
//    }
//  }
//
//  int numDecodings(String s) {
//    int preCnt = 1, cnt = 1;
//    for (int i = 1; i < s.length(); i++) {
//      char hi = s.charAt(i - 1), lo = s.charAt(i);
//      int tmp = cnt;
//      if (hi == '1' || (hi == '2' && lo >= '1' && lo <= '6')) cnt += preCnt;
//      else if (lo == '0') {
//        if (hi != '1' && hi != '2') return 0;
//        cnt = preCnt;
//      }
//      preCnt = tmp;
//    }
//  }
//
//  int candy(int[] ratings) {
//    for (int i = 0; i < len; i++) l[i] = i > 0 && ratings[i] > ratings[i - 1] ? l[i - 1] + 1 : 1;
//    for (int i = len - 1; i > -1; i--) {
//      r = i < len - 1 && ratings[i] > ratings[i + 1] ? r + 1 : 1;
//      minCnt += Math.max(l[i], r);
//    }
//    return minCnt;
//  }
//
//  int superEggDrop(int K, int N) {
//    int[][] dp = new int[K + 1][N + 1];
//    int celling = 0;
//    while (dp[K][celling] < N) {
//      celling += 1;
//      for (int k = 1; k <= K; k++)
//        dp[k][celling] = dp[k][celling - 1] + dp[k - 1][celling - 1] + 1;
//    }
//    return celling;
//  }
//}

class TTree {
  TreeNode buildTree(
          int[] preorder, int preLo, int preHi, Map<Integer, Integer> v2i, int inLo) {
    if (preLo > preHi) return null;
    TreeNode root = new TreeNode(preorder[preLo]);
    int idx = v2i.get(preorder[preLo]), cntL = idx - inLo;
    root.left = buildTree1(preorder, preLo + 1, preLo + cntL, v2i, inLo);
    root.right = buildTree1(preorder, preLo + cntL + 1, preHi, v2i, idx + 1);
    return root;
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

  int pathSumIII(TreeNode root, int targetSum) {
    return dfs1(root, preSum2Cnt, 0, targetSum);
  }

  int dfs1(TreeNode root, Map<Long, Integer> preSum2Cnt, long sum, int target) {
    int cnt = preSum2Cnt.get(sum - target);
    preSum2Cnt.put(sum, preSum2Cnt.get(sum) + 1);
    cnt += dfs14(root.left, preSum2Cnt, sum, target) + dfs14(root.right, preSum2Cnt, sum, target);
    preSum2Cnt.put(sum, preSum2Cnt.get(sum) - 1);
    return cnt;
  }

  int longestIncreasingPath(int[][] matrix) {
    for (int r = 0; r < matrix.length; r++) {
      for (int c = 0; c < matrix[0].length; c++) {
        maxLen = Math.max(maxLen, dfs2(matrix, r, c, lens));
      }
    }
    return maxLen;
  }

  int dfs2(int[][] matrix, int r, int c, int[][] lens) {
    if (lens[r][c] != 0) return lens[r][c];
    lens[r][c] += 1;
    for (int[] dir : DIRECTIONS) {
      int nr = r + dir[0], nc = c + dir[1];
      if (!inArea(matrix, nr, nc) || matrix[nr][nc] <= matrix[r][c]) continue;
      lens[r][c] = Math.max(lens[r][c], dfs3(matrix, nr, nc, lens) + 1);
    }
    return lens[r][c];
  }

  TreeNode treeToDoublyList(TreeNode root) {
    dfs7(root);
    pre.right = head;
    head.left = pre;
  }

  void dfs7(TreeNode root) {
    dfs7(root.left);
    if (pre == null) head = root;
    else pre.right = root;
    root.left = pre;
    pre = root;
    dfs7(root.right);
  }

  void recoverTree(TreeNode root) {
    while (...) {
      // ...
      cur = stack.pollLast();
      if (pre.val > cur.val && n1 == null) n1 = pre;
      if (pre.val > cur.val && n1 != null) n2 = cur;
      pre = cur;
      // ...
    }
  }

  TreeNode insertIntoBST(TreeNode root, int val) {
    while (cur != null) {
      if (val > cur.val) {
        if (cur.right == null) {
          cur.right = new TreeNode(val);
          break;
        }
        cur = cur.right;
      } else {
        if (cur.left == null) {
          cur.left = new TreeNode(val);
          break;
        }
        cur = cur.left;
      }
    }
    return root;
  }

  TreeNode mergeTrees(TreeNode r1, TreeNode r2) {
    while (queue.size() > 0) {
      TreeNode n1 = queue.poll(), n2 = queue.poll();
      n1.val += n2.val;
      if (n1.left != null && n2.left != null) {
        queue.offer(n1.left);
        queue.offer(n2.left);
      }
      if (n1.left == null) n1.left = n2.left;
    }
    return r1;
  }

  boolean flipEquiv(TreeNode r1, TreeNode r2) {
    if (r1 == null && r2 == null) return true;
    if (r1 == null || r2 == null || r1.val != r2.val) return false;
    return (flipEquiv(r1.left, r2.left) && flipEquiv(r1.right, r2.right))
            || (flipEquiv(r1.left, r2.right) && flipEquiv(r1.right, r2.left));
  }

  List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
    collect(root, parents);
    dfs3(root, null, k, parents, vers);
  }

  void collect(TreeNode node, Map<Integer, TreeNode> parents) {
    if (node.left != null) {
      parents.put(node.left.val, node);
      collectParents(node.left, parents);
    }
    //    if (node.right != null) {}
  }

  void dfs3(
          TreeNode n, TreeNode from, int dist, Map<Integer, TreeNode> parents, List<Integer> vers) {
    if (n == null) return;
    if (dist == 0) {
      vers.add(n.val);
      return;
    }
    dist -= 1;
    TreeNode parent = parents.get(n.val);
    if (n.left != from) dfs17(n.left, n, dist, parents, vers);
    if (n.right != from) dfs17(n.right, n, dist, parents, vers);
    if (parent != from) dfs17(parent, n, dist, parents, vers);
  }

  List<String> restoreIpAddresses(String s) {
    bt6(s, new ArrayDeque<>(4), ips, 0, 0);
  }

  void bt6(String s, Deque<String> path, List<String> res, int start, int segCnt) {
    if (start == len) {
      if (segCnt == 4) res.add(String.join(".", path));
      return;
    }
    for (int i = start; i < len; i++) {
      if (!isValidIP(s, start, i) || i = start || len - i > segCnt * 3) continue;
      path.offerLast(s.substring(start, i + 1));
      backtracking6(s, path, res, i + 1, segCnt + 1);
      path.pollLast();
    }
  }

  List<String> letterCombinations(String digits) {
    bt13(digits, new StringBuilder(), res, 0);
  }

  void bt13(String str, StringBuilder path, List<String> res, int start) {
    for (char ch : LetterMap[str.charAt(start) - '0'].toCharArray()) {
      backtracking13(str, path, res, start + 1);
    }
  }

  List<List<String>> partition(String s) {
    boolean[][] isPalindrome = new boolean[len][len];
    for (int i = 0; i < len; i++) {
      collect(s, i, i, isPalindrome);
      collect(s, i, i + 1, isPalindrome);
    }
    bt11(s, new ArrayDeque<>(), paths, 0, isPalindrome);
  }

  void collect(String s, int lo, int hi, boolean[][] isPalindrome) {}

  void backtracking11(
          String s, Deque<String> path, List<List<String>> res, int start, boolean[][] isPld) {
    for (int i = start; i < s.length(); i++) {
      if (!isPld[start][i]) continue;
      path.offerLast(s.substring(start, i + 1));
    }
  }

  boolean exist(char[][] board, String word) {
    for (int r = 0; r < ROW; r++) {
      for (int c = 0; c < COL; c++) {
        if (bt8(board, r, c, chs, 0, recStack)) return true;
      }
    }
    return false;
  }

  boolean bt8(char[][] board, int r, int c, char[] word, int start, boolean[][] recStack) {
    if (start == word.length - 1) return board[r][c] == word[start];
    if (board[r][c] != word[start]) return false;
    recStack[r][c] = true;
    for (int[] dir : DIRECTIONS) {
      int nX = r + dir[0], nY = c + dir[1];
      if (!recStack[nX][nY]
              && inArea(board, nX, nY)
              && backtracking8(board, nX, nY, word, start + 1, recStack)) return true;
    }
    recStack[r][c] = false;
    return false;
  }

  List<String> binaryTreePaths(TreeNode root) {}

  void bt12(TreeNode root, StringBuilder path, List<String> paths) {
    if (root == null) return;
    String add = (path.length() == 0 ? "" : "->") + String.valueOf(root.val);
    if (root.left == null && root.right == null) paths.add(path.toString());
    path.delete(len - addCnt, len);
  }
}

class AArray {
  int[] sortedSquares(int[] nums) {
    for (int i = len - 1; i > -1; i--) {
      int a = nums[lo] * nums[lo], b = nums[hi] * nums[hi];
      if (a <= b) {
        res[i] = b;
        hi -= 1;
      } else {
        res[i] = a;
        lo += 1;
      }
    }
  }

  int minMeetingRooms(int[][] intervals) {
    for (int[] itv : intervals) {
      if (!minHeap.isEmpty() && itv[0] >= minHeap.peek()) minHeap.poll();
      minHeap.offer(itv[1]);
    }
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
    heapify(nums, len - 1);
    for (int i = len - 1; i > 0; i--) {
      swap(nums, 0, i);
      sink(nums, 0, i - 1);
    }
  }

  class MedianFinder {
    private final PriorityQueue<Integer> minHeap = new PriorityQueue<>((a, b) -> a - b),
            maxHeap = new PriorityQueue<>((a, b) -> b - a);

    public void addNum(int n) {
      if (maxHeap.size() > minHeap.size()) {
        if (n < maxHeap.peek()) {
          minHeap.add(maxHeap.poll());
          maxHeap.add(n);
        } else {
          minHeap.add(n);
        }
      } else {
        if (!minHeap.isEmpty() && n > minHeap.peek()) {
          maxHeap.add(minHeap.poll());
          minHeap.add(n);
        } else {
          maxHeap.add(n);
        }
      }
    }

    public double findMedian() {
      return maxHeap.size() > minHeap.size() ? maxHeap.peek() : (maxHeap.peek() + minHeap.peek()) / 2.0;
    }
  }

  private void divideAndCount(int[] nums, int[] tmp, int start, int end) {
    if (start >= end) return;
    int mid = start + (end - start) / 2;
    divideAndCount(nums, tmp, start, mid);
    divideAndCount(nums, tmp, mid + 1, end);
    if (nums[mid] <= nums[mid + 1]) return;
    cnt += mergeAndCount(nums, tmp, start, mid, end);
  }

  private int mergeAndCount(int[] nums, int[] tmp, int start, int mid, int end) {
    if (end > start) System.arraycopy(nums, start, tmp, start, end - start + 1);
    int curCnt = 0, p1 = start, p2 = mid + 1;
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
        curCnt += mid - p1 + 1;
      }
    }
  }

  double findMedianSortedArrays(int[] nums1, int[] nums2) {
    return (getLargestElement(nums1, nums2, top) + getLargestElement(nums1, nums2, topMore)) * 0.5;
  }

  String removeDuplicates(String s) {
    int top = -1;
    for (char ch : chs) {
      if (top > -1 && chs[top] == ch) top -= 1;
      else {
        top += 1;
        chs[top] = ch;
      }
    }
    return String.valueOf(chs, 0, top + 1);
  }

  int removeDuplicates(int[] nums) {
    int write = 0;
    for (int n : nums) {
      if (write >= k && nums[write - k] == n) continue;
      nums[write++] = n;
    }
    return write;
  }

  /****************************二分***************************/
  int getLargestElement(int[] nums1, int[] nums2, int k) {
    while (p1 > -1 && p2 > -1 && ranking > 1) {
      int half = ranking /2,
              newP1 = Math.max(p1 - half, -1) +1,
      if (nums1[newP1] >= nums2[newP2]) {
        ranking -= (p1 - newP1 + 1);
        p1 = newP1 - 1;
      } else {}
    }
    if (p1 == l1) return nums2[p2 - ranking + 1];
    return Math.max(nums1[p1], nums2[p2]);
  }

  int search(int[] nums, int target) {
    while (lo < hi) {
      if (nums[mid] < nums[hi]) {
        if (nums[mid + 1] <= target && target <= nums[hi]) lo = mid + 1;
        else hi = mid;
      } else {
        if (nums[lo] <= target && target <= nums[mid]) hi = mid;
        else lo = mid + 1;
      }
    }
  }

  int findMin(int[] nums) {
    while (lo < hi) {
      if (nums[mid] >= nums[hi]) lo = mid + 1;
      else hi = mid;
    }
  }

  int findPeakElement(int[] nums) {
    while (lo < hi) {
      if (nums[mid] < nums[mid + 1]) lo = mid + 1;
      else hi = mid;
    }
  }

  boolean searchMatrix(int[][] matrix, int target) {
    while (lo < hi) {
      int mid = lo + (hi - lo + 1) / 2;
      if (matrix[mid][0] <= target) lo = mid;
      else hi = mid - 1;
    }
    int[] row = matrix[hi];
    if (row[0] == target) return true;
    if (row[0] > target) return false;
    int c = upperBound(row, target, 0);
    return c == -1 ? false : row[c] == target;
  }

  int kthSmallest(int[][] matrix, int k) {
    int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1];
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (countLte(matrix, mid) < k) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  int countLte(int[][] matrix, int target) {
    int c = 0, r = matrix.length - 1;
    while (-1 < r && c < matrix[0].length) {
      if (matrix[r][c] <= target) {
        cnt += r + 1;
        c += 1;
      } else {
        r -= 1;
      }
    }
  }

  /****************************前缀和***************************/

  int maxSubArray(int[] nums) {
    for (int n : nums) {
      if (preSum + n > n) {
        preSum += n;
      } else {
        preSum = n;
      }
      maxSum = Math.max(maxSum, preSum);
    }
    return maxSum;
  }

  int subarraySum(int[] nums, int k) {
    Map<Integer, Integer> sum2Cnt = new HashMap<>();
    sum2Cnt.put(0, 1);
    for (int n : nums) {
      preSum += n;
      cnt += sum2Cnt.get(preSum - k);
      sum2Cnt.put(preSum, 1 + sum2Cnt.get(preSum));
    }
    return cnt;
  }

  int findMaxLength(int[] nums) {
    Map<Integer, Integer> sum2FirstIdx = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
      preSum += nums[i] == 0 ? -1 : 1;
      if (!sum2FirstIdx.containsKey(preSum)) sum2FirstIdx.put(preSum, i);
      maxLen = Math.max(maxLen, i - sum2FirstIdx.get(preSum));
    }
    return maxLen;
  }

  int subarraysDivByK(int[] nums, int k) {
    HashMap<Integer, Integer> remainder2Cnt = new HashMap<>();
    remainder2Cnt.put(0, 1);
    for (int n : nums) {
      preSum += n;
      int remainder = (preSum % k + k) % k;
      int curCnt = remainder2Cnt.get(remainder);
      remainder2Cnt.put(remainder, curCnt + 1);
      cnt += curCnt;
    }
    return cnt;
  }

  int shortestSubarray(int[] nums, int k) {
    for (int i = 0; i < preSum.length; i++) {
      long sum = preSum[i];
      while (!mq.isEmpty() && sum <= preSum[mq.peekLast()]) mq.pollLast();
      while (!mq.isEmpty() && sum >= k + preSum[mq.peekFirst()])
        minLen = Math.min(minLen, i - mq.pollFirst());
      mq.offerLast(i);
    }
  }

  boolean checkSubarraySum(int[] nums, int target) {
    Set<Integer> visted = new HashSet<>();
    for (int i = 2; i <= len; i++) {
      visted.add(preSum[i - 2] % target);
      if (visted.contains(preSum[i] % target)) return true;
    }
  }

  /****************************字典序***************************/

  void nextPermutation(int[] nums) {
    while (peak > 0) {
      if (nums[peak - 1] < nums[peak]) {
        Arrays.sort(nums, peak, len);
        break;
      }
      peak -= 1;
    }
    for (int i = peak; i < len; i++) {
      if (nums[peak - 1] >= nums[i]) continue;
      swap(nums, peak - 1, i);
      return;
    }
    Arrays.sort(nums);
  }

  int nextGreaterElement(int n) {
    while (peak > -1) {
      if (chs[peak] < chs[peak - 1]) break;
      peak -= 1;
    }
    if (peak == -1) return -1;
    for (int i = len - 1; i > peak; i--) {
      if (chs[peak - 1] >= chs[i]) continue;
      swap(chs, peak, i);
      break;
    }
    revserse(chs, peak + 1, len - 1);
  }

  String largestNumber(int[] nums) {
    // nums -> String[]
    strs.sort((s1, s2) -> (s2 + s1).compareTo(s1 + s2));
    // String[] -> String
  }

  int maximumSwap(int num) {
    for (int i = 0; i < chs.length; i++) lastIdxes[chs[i] - '0'] = i;
    for (int i = 0; i < chs.length; i++) {
      for (int d = 9; d > chs[i] - '0'; d--) {
        if (lastIdxes[d] <= i) continue; // 位
        swap(chs, i, lastIdxes[d]);
        return Integer.parseInt(chs.toString());
      }
    }
    return num;
  }

  public int findKthNumber(int n, int k) {
    while (cnt < k) {
      int curCnt = count(prefix, n);
      if (cnt + curCnt >= k) {
        prefix *= 10;
        cnt += 1;
      } else {
        prefix += 1;
        cnt += curCnt;
      }
    }
    return prefix;
  }

  int count(int lo, int hi) {
    long cur = lo, nxt = lo + 1;
    while (cur <= hi) {
      cnt += Math.min(hi + 1, nxt) - cur;
      cur *= 10;
      nxt *= 10;
    }
  }
}

class SString {
  String addStrings(String num1, String num2) {
    while (p1 > -1 || p2 > -1 || carry != 0) {
      int n1 = p1 < 0 ? 0 : num1.charAt(p1) - '0',
              n2 = p2 < 0 ? 0 : num2.charAt(p2) - '0',
              tmp = n1 + n2 + carry;
      sum.append(getChar(tmp % BASE));
      carry = tmp / BASE;
      p1 -= 1;
      p2 -= 1;
    }
    return sum.reverse().toString();
  }

  String multiply(String num1, String num2) {
    for (int p1 = l1 - 1; p1 > -1; p1--) {
      int n1 = num1.charAt(p1) - '0';
      for (int p2 = l2 - 1; p2 > -1; p2--) {
        int n2 = num2.charAt(p2) - '0', sum = product[p1 + p2 + 1] + n1 * n2;
        product[p1 + p2 + 1] = sum % BASE;
        product[p1 + p2] += sum / BASE;
      }
    }
    int idx = product[0] == 0 ? 1 : 0;
    StringBuilder res = new StringBuilder();
    while (idx < product.length) {
      res.append(product[idx]);
      idx += 1;
    }
  }

  int compareVersion(String v1, String v2) {
    while (p1 < l1 || p2 < l2) {
      int n1 = 0, n2 = 0;
      while (p1 < l1 && v1.charAt(p1) != '.') {
        n1 = n1 * 10 + v1.charAt(p1) - '0';
        p1 += 1;
      }
      p1 += 1;
      if (n1 != n2) return n1 > n2 ? 1 : -1;
    }
  }

  int longestOnes(int[] nums, int k) {
    while (hi < nums.length) {
      if (nums[hi] == 0) k -= 1;
      while (k < 0) {
        if (nums[lo] == 0) k += 1;
        lo += 1;
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
      hi += 1;
    }
  }

  int[] maxSlidingWindow(int[] nums, int k) {
    for (int i = 0; i < nums.length; i++) {
      while (mq.size() > 0 && mq.peekLast() < n) mq.pollLast();
      mq.offerLast(n);
      if (i < k - 1) continue;
      int outIdx = i - k + 1, winMax = mq.peekFirst();
      winMaxes[outIdx] = winMax;
      if (mq.size() > 0 && nums[outIdx] == winMax) mq.pollFirst();
    }
  }

  int longestSubarray(int[] nums, int limit) {
    while (hi < nums.length) {
      int add = nums[hi];
      hi += 1;
      while (maxMQ.size() > 0 && add > maxMQ.peekLast()) maxMQ.pollLast();
      maxMQ.offerLast(add);
      while (minMQ.size() > 0 && add < minMQ.peekLast()) minMQ.pollLast();
      minMQ.offerLast(add);
      while (maxMQ.peekFirst() - minMQ.peekFirst() > limit) {
        int out = nums[lo];
        lo += 1;
        if (maxMQ.peekFirst() == out) maxMQ.pollFirst();
        if (minMQ.peekFirst() == out) minMQ.pollFirst();
      }
      maxLen = Math.max(maxLen, hi - lo);
    }
  }

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

  int calculate(String s) {
    char op = '+';
    for (int i = 0; i < len; i++) {
      char ch = chs[i];
      if (Character.isDigit(ch)) n = n * 10 + (ch - '0');
      if ((!Character.isDigit(ch) && ch != ' ') || i == len - 1) {
        if (op == '+') numStack.offerLast(n);
        if (op == '-') numStack.offerLast(-n);
        if (op == '*') numStack.offerLast(numStack.pollLast() * n);
        if (op == '/') numStack.offerLast(numStack.pollLast() / n);
        n = 0;
        op = ch;
      }
    }
    int sum = 0;
    for (int num : numStack) sum += num;
  }

  int[] countLetter(char[] chs) {
    opStack.offerLast(sign);
    for (int i = 0; i < chs.length; i++) {
      char ch = chs[i];
      if (ch == '(') opStack.offerLast(sign);
      if (ch == ')') opStack.pollLast();
      if (ch == '+') sign = opStack.peekLast();
      if (ch == '-') sign = -opStack.peekLast();
      if (ch >= '0' && ch <= '9') {
        long num = 0;
        while (i < chs.length && Character.isDigit(chs[i])) {
          num = num * 10 + chs[i] - '0';
          i += 1;
        }
        i -= 1;
        res += sign * num;
      }
    }
  }

  boolean validateStackSequences(int[] pushed, int[] popped) {
    for (int add : pushed) {
      pushed[stTop] = add;
      stTop += 1;
      while (stTop != 0 && pushed[stTop - 1] == popped[popIdx]) {
        stTop -= 1;
        popIdx += 1;
      }
    }
    return stTop == 0;
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

  int romanToInt(String s) {
    for (int i = 0; i < chs.length; i++) {
      int add = mapping.get(chs[i]);
      // IV 与 VI 否则常规的做法是累加即可
      if (i < chs.length - 1 && add < mapping.get(chs[i + 1])) sum -= add;
      else sum += add;
    }
  }

  String convertToTitle(int cn) {
    int cur = cn - 1;
    while (cur > 0) {
      res.append((char) (cur % 26 + 'A'));
      cur = cur / 26 - 1;
    }
    return res.reverse().toString();
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

  String removeKdigits(String num, int k) {
    for (char ch : num.toCharArray()) {
      while (k > 0 && ms.length() > 0 && ms.charAt(ms.length() - 1) > ch) {
        ms.deleteCharAt(ms.length() - 1);
        k -= 1;
      }
      if (ch == '0' && ms.length() == 0) continue; // 避免前导零
      ms.append(ch);
    }
    String str = ms.substring(0, Math.max(ms.length() - k, 0));
  }

  int maximalRectangle(char[][] matrix) {
    int[] hs = new int[len + 2];
    for (char[] rows : matrix) {
      for (int i = 0; i < hs.length; i++) {
        if (i > 0 && i < len + 1) { // 哨兵内
          if (rows[i - 1] == '1') hs[i] += 1;
          else hs[i] = 0;
        }
        while (!ms.isEmpty() && hs[ms.peekLast()] > hs[i]) {
          int cur = ms.pollLast(), pre = ms.peekLast();
          maxArea = Math.max(maxArea, (i - pre - 1) * hs[cur]);
        }
        ms.offerLast(i);
      }
    }
    return maxArea;
  }
}