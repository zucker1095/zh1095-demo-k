package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集树相关，BFS 参下，其余包括
 *
 * <p>关于 Java 模拟 stack 的选型
 * https://qastack.cn/programming/6163166/why-is-arraydeque-better-than-linkedlist
 *
 * <p>TODO 前序，尝试均改为迭代
 *
 * <p>中序，基本即是 BST
 *
 * <p>后序，统计相关 & 递归，参下
 *
 * <p>扩展大部分与记录路径相关
 *
 * @author cenghui
 */
public class TTree {
  private int res3 = 0;

  /**
   * 中序遍历迭代，注意区分遍历 & 处理两个 step
   *
   * <p>模板参考
   * https://leetcode-cn.com/problems/binary-tree-postorder-traversal/solution/zhuan-ti-jiang-jie-er-cha-shu-qian-zhong-hou-xu-2/
   *
   * <p>思维参考
   * https://leetcode-cn.com/problems/binary-tree-paths/solution/tu-jie-er-cha-shu-de-suo-you-lu-jing-by-xiao_ben_z/
   *
   * @param root the root
   * @return the list
   */
  public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.addLast(cur); // 1.traverse
        cur = cur.left;
      }
      cur = stack.removeLast(); // 2.handle
      res.add(cur.val);
      cur = cur.right;
    }
    return res;
  }

  /**
   * 二叉树的后序遍历
   *
   * @param root the root
   * @return list list
   */
  public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        res.add(cur.val);
        stack.addLast(cur);
        cur = cur.right; // left pre
      }
      cur = stack.removeLast();
      cur = cur.left; // right pre
    }
    Collections.reverse(res);
    return res;
  }

  /**
   * 从前序与中序遍历序列构造二叉树，题设元素唯一，否则，存在多棵树
   *
   * <p>扩展1，根据前序和中序，输出后序，不能构造树，参下 annotate
   *
   * <p>扩展2，给一个随机数组，生成相应的二叉搜索树，先排序，参下「将有序数组转换为二叉搜索树」
   *
   * <p>扩展3，从中序和后序构造，参下
   *
   * @param preorder the preorder
   * @param inorder the inorder
   * @return the tree node
   */
  public TreeNode buildTree(int[] preorder, int[] inorder) {
    Map<Integer, Integer> idxByValInorder = new HashMap<>();
    for (int i = 0; i < preorder.length; i++) {
      idxByValInorder.put(inorder[i], i);
    }
    return buildTree1(preorder, 0, preorder.length - 1, idxByValInorder, 0);
  }

  private TreeNode buildTree1(
      int[] preorder, int preLo, int preHi, Map<Integer, Integer> idxByValInorder, int inLo) {
    //    if (preLo == preHi) {
    //      postorder.add(preorder[preLo]);
    //      return;
    //    }
    if (preLo > preHi) return null;
    TreeNode root = new TreeNode(preorder[preLo]);
    int idx = idxByValInorder.get(preorder[preLo]);
    int countLeft = idx - inLo;
    root.left = buildTree1(preorder, preLo + 1, preLo + countLeft, idxByValInorder, inLo);
    root.right = buildTree1(preorder, preLo + countLeft + 1, preHi, idxByValInorder, idx + 1);
    //    postorder.add(preorder[preLo]);
    return root;
  }

  // 从中序与后序遍历序列构造二叉树
  private TreeNode buildTree2(
      int[] postrorder, int postLo, int postHi, Map<Integer, Integer> idxByValInorder, int inLo) {
    if (postLo > postHi) return null;
    TreeNode root = new TreeNode(postrorder[postHi]);
    int idx = idxByValInorder.get(postrorder[postHi]);
    int countLeft = idx - inLo;
    root.left = buildTree2(postrorder, postLo, postLo + countLeft - 1, idxByValInorder, inLo);
    root.right = buildTree2(postrorder, postLo + countLeft, postHi - 1, idxByValInorder, idx + 1);
    return root;
  }

  /**
   * 路径总和，从根出发要求达到叶，前序遍历，记录路径则参下「路径总和II」回溯
   *
   * @param root the root
   * @param targetSum the target sum
   * @return boolean boolean
   */
  public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root == null) return false;
    if (root.left == null && root.right == null) return targetSum == root.val;
    // 剪枝
    return hasPathSum(root.left, targetSum - root.val)
        || hasPathSum(root.right, targetSum - root.val);
  }

  /**
   * 合并二叉树，前序
   *
   * @param r1 the r 1
   * @param r2 the r 2
   * @return tree node
   */
  public TreeNode mergeTrees(TreeNode r1, TreeNode r2) {
    if (r1 == null || r2 == null) return r1 == null ? r2 : r1;
    r1.val += r2.val;
    r1.left = mergeTrees(r1.left, r2.left);
    r1.right = mergeTrees(r1.right, r2.right);
    return r1;
  }

  /**
   * 另一棵树的子树
   *
   * <p>特判匹配树 & 主树为空两种情况，isSameTree 中的两处特判可以去除，因为匹配树 & 主树均非空
   *
   * <p>TODO 面试题 04.10.检查子树 找子结构 t572 找子树 t1376 找链表
   *
   * @param root the root
   * @param subRoot the sub root
   * @return boolean boolean
   */
  public boolean isSubtree(TreeNode root, TreeNode subRoot) {
    if (subRoot == null) return true;
    if (root == null) return false;
    return isSubtree(root.left, subRoot)
        || isSubtree(root.right, subRoot)
        || isSameTree(root, subRoot);
  }

  /**
   * 翻转等价二叉树
   *
   * @param root1 the root 1
   * @param root2 the root 2
   * @return boolean
   */
  public boolean flipEquiv(TreeNode root1, TreeNode root2) {
    if (root1 == root2) return true;
    if (root1 == null || root2 == null || root1.val != root2.val) return false;
    return flipEquiv(root1.left, root2.left) && flipEquiv(root1.right, root2.right)
        || flipEquiv(root1.left, root2.right) && flipEquiv(root1.right, root2.left);
  }

  /**
   * 相同的树，前序，迭代选用 bfs
   *
   * @param p the p
   * @param q the q
   * @return boolean boolean
   */
  public boolean isSameTree(TreeNode p, TreeNode q) {
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(p);
    queue.offer(q);
    while (!queue.isEmpty()) {
      p = queue.poll();
      q = queue.poll();
      if (p == null && q == null) continue;
      if (p == null || q == null || p.val != q.val) {
        return false;
      }
      // 顺序
      queue.offer(p.left);
      queue.offer(q.left);
      queue.offer(p.right);
      queue.offer(q.right);
    }
    return true;
  }

  /**
   * 求根结点到叶子结点数字之和，bfs 维护两个队列 / 前序
   *
   * @param root the root
   * @return int int
   */
  public int sumNumbers(TreeNode root) {
    return bfs(root);
    //    dfs12(root, 0);
    //    return res3;
  }

  private int bfs(TreeNode root) {
    if (root == null) return 0;
    int sum = 0;
    Queue<TreeNode> nodeQueue = new LinkedList<>();
    Queue<Integer> numQueue = new LinkedList<>();
    nodeQueue.offer(root);
    numQueue.offer(root.val);
    while (!nodeQueue.isEmpty()) {
      TreeNode node = nodeQueue.poll();
      int num = numQueue.poll();
      TreeNode left = node.left, right = node.right;
      if (left == null && right == null) {
        sum += num;
        continue;
      }
      if (left != null) {
        nodeQueue.offer(left);
        numQueue.offer(num * 10 + left.val);
      }
      if (right != null) {
        nodeQueue.offer(right);
        numQueue.offer(num * 10 + right.val);
      }
    }
    return sum;
  }

  private void dfs12(TreeNode root, int path) {
    if (root == null) return;
    // 题设不会越界
    int cur = path * 10 + root.val;
    if (root.left == null && root.right == null) {
      res3 += cur;
      return;
    }
    dfs12(root.left, cur);
    dfs12(root.right, cur);
  }

  /**
   * 二叉树的下一个结点，给定一棵树中任一结点，返回其中序遍历顺序的下一个结点，提供结点的 next 指向父结点
   *
   * <p>参考 https://mp.weixin.qq.com/s/yewlHvHSilMsrUMFIO8WAA
   *
   * @param node the node
   * @return next
   */
  public _TreeNode getNext(_TreeNode node) {
    // 如果有右子树，则找右子树的最左结点
    if (node.right != null) {
      _TreeNode cur = node.right;
      while (cur.left != null) {
        cur = cur.left;
      }
      return cur;
    }
    // 否则，找第一个当前结点是父结点左孩子的结点
    _TreeNode cur = node;
    while (cur.next != null) {
      if (cur.next.left.equals(cur)) {
        return cur.next;
      }
      cur = cur.next;
    }
    // 退到根结点仍没找到
    return null;
  }

  private class _TreeNode {
    /** The Left. */
    _TreeNode left,
        /** The Right. */
        right,
        /** The Next. */
        next;
  }

  /** 二叉树的序列化与反序列化，前序 */
  public class Codec {

    /**
     * Serialize string.
     *
     * @param root the root
     * @return the string
     */
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
      return root == null
          ? "null"
          : root.val + "," + serialize(root.left) + "," + serialize(root.right);
    }

    /**
     * Deserialize tree node.
     *
     * @param data the data
     * @return the tree node
     */
    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
      Queue<String> queue = new LinkedList<>(Arrays.asList(data.split(",")));
      return dfs(queue);
    }

    private TreeNode dfs(Queue<String> queue) {
      String val = queue.poll();
      if ("null".equals(val)) return null;
      TreeNode root = new TreeNode(Integer.parseInt(val));
      root.left = dfs(queue);
      root.right = dfs(queue);
      return root;
    }
  }
}

/**
 * 回溯，前序与后序结合，遵从如下规范
 *
 * <p>入参顺序为 selection, path, res(if need), ...args
 *
 * <p>按照子组列的顺序，建议按照表格记忆
 */
class BBacktracking extends DDFS {
  /**
   * 路径总和II，从根出发要求达到叶，需要记录路径
   *
   * @param root the root
   * @param targetSum the target sum
   * @return list list
   */
  public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
    List<List<Integer>> res = new ArrayList<>();
    backtracking0(root, new ArrayDeque<>(), res, targetSum);
    return res;
  }

  private void backtracking0(
      TreeNode root, Deque<Integer> path, List<List<Integer>> res, int targetSum) {
    if (root == null) return;
    path.addLast(root.val);
    if (targetSum - root.val == 0 && root.left == null && root.right == null) {
      // path 全局唯一，须做拷贝
      res.add(new ArrayList<>(path));
      // return 前须重置
      path.removeLast();
      return;
    }
    backtracking0(root.left, path, res, targetSum - root.val);
    backtracking0(root.right, path, res, targetSum - root.val);
    // 递归完成以后，必须重置变量
    path.removeLast();
  }

  /**
   * 路径总和III，返回路径总数，但从任意点出发，回溯 & 前缀和
   *
   * <p>node.val:从该点出发满足的路径总数，则任两点不会有重复的路径
   *
   * @param root the root
   * @param targetSum the target sum
   * @return int int
   */
  public int pathSumIII(TreeNode root, int targetSum) {
    Map<Long, Integer> prefix = new HashMap<>() {};
    prefix.put(0L, 1); // base case
    return backtracking9(root, prefix, 0, targetSum);
  }

  private int backtracking9(TreeNode root, Map<Long, Integer> preSum, long cur, int targetSum) {
    if (root == null) return 0;
    cur += root.val;
    int res = preSum.getOrDefault(cur - targetSum, 0);
    preSum.put(cur, preSum.getOrDefault(cur, 0) + 1);
    res +=
        backtracking9(root.left, preSum, cur, targetSum)
            + backtracking9(root.right, preSum, cur, targetSum);
    preSum.put(cur, preSum.getOrDefault(cur, 0) - 1);
    return res;
  }

  /**
   * 子集
   *
   * @param nums the nums
   * @return the list
   */
  public List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums.length == 0) return res;
    backtracking1(nums, new ArrayDeque<>(), res, 0);
    return res;
  }

  private void backtracking1(int[] nums, Deque<Integer> path, List<List<Integer>> res, int start) {
    res.add(new ArrayList<>(path));
    for (int i = start; i < nums.length; i++) {
      path.addLast(nums[i]);
      backtracking1(nums, path, res, i + 1);
      path.removeLast();
    }
  }

  /**
   * 组合总和I
   *
   * @param candidates the candidates
   * @param target the target
   * @return the list
   */
  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    if (candidates.length == 0) return res;
    Arrays.sort(candidates);
    backtracking2(candidates, new ArrayDeque<>(), res, 0, target);
    return res;
  }

  private void backtracking2(
      int[] candidates, Deque<Integer> path, List<List<Integer>> res, int start, int target) {
    if (target == 0) res.add(new ArrayList<>(path));
    for (int i = start; i < candidates.length; i++) {
      if (candidates[i] > target) break;
      path.addLast(candidates[i]);
      backtracking2(candidates, path, res, i, target - candidates[i]);
      path.removeLast();
    }
  }

  /**
   * 组合总和II
   *
   * @param candidates the candidates
   * @param target the target
   * @return the list
   */
  public List<List<Integer>> combinationSum2(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    if (candidates.length == 0) return res;
    Arrays.sort(candidates);
    backtracking3(candidates, new ArrayDeque<>(), res, 0, target);
    return res;
  }

  private void backtracking3(
      int[] candidates, Deque<Integer> path, List<List<Integer>> res, int start, int target) {
    if (target == 0) res.add(new ArrayList<>(path));
    for (int i = start; i < candidates.length; i++) {
      if (candidates[i] > target) break;
      if (i > start && candidates[i - 1] == candidates[i]) continue;
      path.addLast(candidates[i]);
      backtracking3(candidates, path, res, i + 1, target - candidates[i]);
      path.removeLast();
    }
  }

  /**
   * 全排列I
   *
   * @param nums the nums
   * @return the list
   */
  public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums.length == 0) return res;
    backtracking4(nums, new ArrayDeque<>(), res, new boolean[nums.length]);
    return res;
  }

  private void backtracking4(
      int[] nums, Deque<Integer> path, List<List<Integer>> res, boolean[] visited) {
    if (path.size() == nums.length) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if (visited[i]) continue;
      visited[i] = true;
      path.addLast(nums[i]);
      backtracking4(nums, path, res, visited);
      path.removeLast();
      visited[i] = false;
    }
  }

  /**
   * 全排列II
   *
   * @param nums the nums
   * @return the list
   */
  public List<List<Integer>> permuteUnique(int[] nums) {
    List<List<Integer>> res = new ArrayList<>();
    if (nums.length == 0) return res;
    Arrays.sort(nums);
    backtracking5(nums, new ArrayDeque<>(), res, new boolean[nums.length]);
    return res;
  }

  private void backtracking5(
      int[] nums, Deque<Integer> path, List<List<Integer>> res, boolean[] visited) {
    if (path.size() == nums.length) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if (visited[i] || (i > 0 && nums[i] == nums[i - 1] && !visited[i])) continue;
      visited[i] = true;
      path.addLast(nums[i]);
      backtracking5(nums, path, res, visited);
      path.removeLast();
      visited[i] = false;
    }
  }

  /**
   * 括号生成，括号相关的参考「最长有效括号」与「有效的括号」
   *
   * @param n the n
   * @return list list
   */
  public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    // 需要特判，否则入串
    if (n <= 0) return res;
    backtracking7(n, n, "", res);
    return res;
  }

  // 此处的可选集为左右括号的剩余量，因为每次尝试，都使用新的字符串，所以无需显示回溯
  private void backtracking7(int left, int right, String path, List<String> res) {
    if (left == 0 && right == 0) {
      res.add(path);
      return;
    }
    if (left > right) return;
    if (left > 0) backtracking7(left - 1, right, path + "(", res);
    if (right > 0) backtracking7(left, right - 1, path + ")", res);
  }

  /**
   * 单词搜索
   *
   * @param board the board
   * @param word the word
   * @return boolean boolean
   */
  public boolean exist(char[][] board, String word) {
    int rows = board.length, cols = board[0].length;
    boolean[][] visited = new boolean[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (backtracking8(board, i, j, word, 0, visited)) return true;
      }
    }
    return false;
  }

  private boolean backtracking8(
      char[][] board, int r, int c, String word, int start, boolean[][] visited) {
    if (start == word.length() - 1) {
      return board[r][c] == word.charAt(start);
    }
    if (board[r][c] != word.charAt(start)) {
      return false;
    }
    visited[r][c] = true;
    for (int[] dir : DIRECTIONS) {
      int newX = r + dir[0], newY = c + dir[1];
      if (!visited[newX][newY]
          && inArea(board, newX, newY)
          && backtracking8(board, newX, newY, word, start + 1, visited)) {
        return true;
      }
    }
    visited[r][c] = false;
    return false;
  }

  /**
   * 分割回文串
   *
   * <p>1.预处理所有子串的回文情况
   *
   * <p>2.暴力回溯
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/palindrome-partitioning/solution/hui-su-you-hua-jia-liao-dong-tai-gui-hua-by-liweiw/
   *
   * @param s the s
   * @return list
   */
  public List<List<String>> partition(String s) {
    List<List<String>> res = new ArrayList<>();
    backtracking11(s, new ArrayDeque<String>(), res, 0, parseString(s));
    return res;
  }

  // 预处理 dp[i][j] 表示 s[i][j] 是否是回文
  // 状态转移，在 s[i] == s[j] 的时候，dp[i][j] 参考 dp[i + 1][j - 1]
  private boolean[][] parseString(String s) {
    int len = s.length();
    char[] chs = s.toCharArray();
    boolean[][] dp = new boolean[len][len];
    for (int hi = 0; hi < len; hi++) {
      // 双指针碰撞表示一个字符的时候也需要判断
      for (int lo = 0; lo <= hi; lo++) {
        dp[lo][hi] = chs[lo] == chs[hi] && (hi - lo <= 2 || dp[lo + 1][hi - 1]);
      }
    }
    return dp;
  }

  private void backtracking11(
      String s, Deque<String> path, List<List<String>> res, int idx, boolean[][] dp) {
    if (idx == s.length()) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = idx; i < s.length(); i++) {
      if (!dp[idx][i]) continue;
      path.addLast(s.substring(idx, i + 1));
      backtracking11(s, path, res, i + 1, dp);
      path.removeLast();
    }
  }

  /**
   * 复原IP地址，特判多
   *
   * @param s the s
   * @return list list
   */
  public List<String> restoreIpAddresses(String s) {
    List<String> res = new ArrayList<>();
    // 特判
    if (s.length() > 12 || s.length() < 4) {
      return res;
    }
    backtracking6(s, new ArrayDeque<>(4), res, 0, 4);
    return res;
  }

  private void backtracking6(
      String s, Deque<String> path, List<String> res, int start, int residue) {
    if (start == s.length()) {
      if (residue == 0) res.add(String.join(".", path));
      return;
    }
    for (int i = start; i < start + 3 && i < s.length(); i++) {
      if (residue * 3 < s.length() - i || !isValidIpSegment(s, start, i)) {
        continue;
      }
      path.addLast(s.substring(start, i + 1));
      backtracking6(s, path, res, i + 1, residue - 1);
      path.removeLast();
    }
  }

  private boolean isValidIpSegment(String s, int lo, int hi) {
    int res = 0;
    if (hi > lo && s.charAt(lo) == '0') {
      return false;
    }
    while (lo <= hi) {
      res = res * 10 + s.charAt(lo) - '0';
      lo += 1;
    }
    return res >= 0 && res <= 255;
  }

  /**
   * 验证IP地址
   *
   * <p>TODO
   *
   * @param queryIP
   * @return
   */
  // public String validIPAddress(String queryIP) {}

  /**
   * 解数独，暴力回溯
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/sudoku-solver/solution/37-jie-shu-du-hui-su-sou-suo-suan-fa-xiang-jie-by-/
   *
   * @param board the board
   */
  public void solveSudoku(char[][] board) {
    backtracking10(board);
  }

  // 1.跳过原始数字
  // 2.位置放 k 是否合适，是则，找到合适一组立刻返回
  // 3.九个数都试完，说明该棋盘无解
  private boolean backtracking10(char[][] board) {
    for (int y = 0; y < 9; y++) {
      for (int x = 0; x < 9; x++) {
        if (board[y][x] != '.') continue;
        for (char k = '1'; k <= '9'; k++) {
          if (!isValidSudoku(y, x, k, board)) continue;
          board[y][x] = k;
          if (backtracking10(board)) return true;
          board[y][x] = '.';
        }
        return false;
      }
    }
    // 遍历完没有返回 false，说明找到合适棋盘位置
    return true;
  }

  // 判断棋盘是否合法有如下三个维度，同行，同列，九宫格里是否重复
  private boolean isValidSudoku(int row, int col, char val, char[][] board) {
    for (int i = 0; i < 9; i++) {
      if (board[row][i] == val) return false;
    }
    for (int j = 0; j < 9; j++) {
      if (board[j][col] == val) return false;
    }
    int startRow = (row / 3) * 3, startCol = (col / 3) * 3;
    for (int i = startRow; i < startRow + 3; i++) {
      for (int j = startCol; j < startCol + 3; j++) {
        if (board[i][j] == val) return false;
      }
    }
    return true;
  }
}

/**
 * 深度优先搜索
 *
 * <p>对于树，按照遍历的次序，dfs 即选型前序遍历或后序，而回溯相当于同时前序与后序
 *
 * <p>回溯 & dfs 框架基本一致，但前者适用 tree 这类不同分支互不连通的结构，而后者更适合 graph 这类各个分支都可能连通的
 *
 * <p>因此后者为避免环路，不需要回溯，比如下方 grid[i][j]=2 后不需要再恢复
 */
class DDFS {
  /** The Directions. */
  protected final int[][] DIRECTIONS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  // 二叉树中所有距离为 k 的结点，结果集
  private final List<Integer> res5 = new ArrayList<>();
  // 二叉树中所有距离为 k 的结点，目标结点的父
  private TreeNode parent;

  /**
   * 岛屿数量
   *
   * <p>参考
   * https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/
   *
   * @param grid the grid
   * @return int int
   */
  public int numIslands(char[][] grid) {
    int res = 0;
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] != '1') continue;
        dfs1(grid, i, j);
        res += 1;
      }
    }
    return res;
  }

  private void dfs1(char[][] grid, int r, int c) {
    if (!inArea(grid, r, c) || grid[r][c] == '0') {
      return;
    }
    grid[r][c] = '0';
    for (int[] dir : DIRECTIONS) {
      dfs1(grid, r + dir[0], c + dir[1]);
    }
  }

  /**
   * 坐标未越界
   *
   * @param board the board
   * @param i the
   * @param j the j
   * @return the boolean
   */
  protected boolean inArea(char[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }

  /**
   * 岛屿的最大面积
   *
   * @param grid the grid
   * @return int int
   */
  public int maxAreaOfIsland(int[][] grid) {
    int res = 0;
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[i].length; j++) {
        if (grid[i][j] != 1) continue;
        res = Math.max(res, dfs2(grid, i, j));
      }
    }
    return res;
  }

  private int dfs2(int[][] grid, int r, int c) {
    if (!inArea(grid, r, c) || grid[r][c] == 0) return 0;
    grid[r][c] = 0;
    int res = 1;
    for (int[] dir : DIRECTIONS) {
      res += dfs2(grid, r + dir[0], c + dir[1]);
    }
    return res;
  }

  /**
   * Num distinct islands int.
   *
   * @param grid the grid
   * @return the int
   */
  public int numDistinctIslands(int[][] grid) {
    Set<String> res = new HashSet<>();
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] != 1) continue;
        StringBuilder path = new StringBuilder();
        dfs5(grid, path, i, j, i, j);
        res.add(path.toString());
      }
    }
    return res.size();
  }

  private void dfs5(int[][] grid, StringBuilder path, int x, int y, int preX, int preY) {
    if (!inArea(grid, x, y) || grid[x][y] == 0) return;
    grid[x][y] = 0;
    path.append(x - preX); // 记录相对横坐标
    path.append(y - preY); // 记录相对纵坐标
    for (int[] dir : DIRECTIONS) {
      dfs5(grid, path, x + dir[0], y + dir[1], preX, preY);
    }
  }

  /**
   * 矩阵中的最长递增路径
   *
   * @param matrix the matrix
   * @return int int
   */
  public int longestIncreasingPath(int[][] matrix) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
      return 0;
    }
    int rows = matrix.length, cols = matrix[0].length;
    int[][] memo = new int[rows][cols];
    int res = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        res = Math.max(res, dfs3(matrix, i, j, memo));
      }
    }
    return res;
  }

  private int dfs3(int[][] matrix, int r, int c, int[][] memo) {
    if (memo[r][c] != 0) return memo[r][c];
    memo[r][c] += 1;
    for (int[] dir : DIRECTIONS) {
      int newR = r + dir[0], newC = c + dir[1];
      if (inArea(matrix, newR, newC) && matrix[newR][newC] <= matrix[r][c]) {
        memo[r][c] = Math.max(memo[r][c], dfs3(matrix, newR, newC, memo) + 1);
      }
    }
    return memo[r][c];
  }

  private boolean inArea(int[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }

  /**
   * 二叉树中所有距离为k的结点，掌握 DFS 分割即可
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/all-nodes-distance-k-in-binary-tree/solution/gai-bian-shu-de-xing-zhuang-c-si-lu-dai-ma-by-lhrs/
   *
   * @param root the root
   * @param target the target
   * @param k the k
   * @return list list
   */
  public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
    // 首先把树分为两棵，分别以目标结点及其父结点为根
    dfs4(null, root, target);
    // 搜索以目标结点为根的中树深度为 k 的结点
    collect(target, k);
    // 搜索以目标结点父为根的树，、深度为 k-1 的结点
    collect(parent, k - 1);
    return res5;
  }

  // 分割结点
  private Boolean dfs4(TreeNode pre, TreeNode cur, TreeNode target) {
    if (cur == null) return false;
    // 如果搜到了目标结点，那么它父就是新树的根
    if (cur == target) {
      parent = pre;
      return true;
    }
    // 如果我成了左儿子的儿子，那我的父就是我的新的左儿子
    if (dfs4(cur, cur.left, target)) {
      cur.left = pre;
      return true;
    }
    // 如果我成了我右儿子的儿子，那我的父就是我的新的右儿子
    if (dfs4(cur, cur.right, target)) {
      cur.right = pre;
      return true;
    }
    // 递归的时候返回父
    return false;
  }

  // 搜索以 k 为根结点的树，其第 k 层所有结点
  private void collect(TreeNode root, int k) {
    if (root == null) return;
    if (k == 0) {
      res5.add(root.val);
      return;
    }
    collect(root.left, k - 1);
    collect(root.right, k - 1);
  }

  /**
   * 二叉树的所有路径，前序
   *
   * @param root the root
   * @return list
   */
  public List<String> binaryTreePaths(TreeNode root) {
    List<String> res = new ArrayList<>();
    dfs13(root, "", res);
    return res;
  }

  private void dfs13(TreeNode root, String path, List<String> res) {
    // 如果为空，直接返回
    if (root == null) return;
    // 如果是叶子结点，说明找到了一条路径，把它加入到res中
    if (root.left == null && root.right == null) {
      res.add(path + root.val);
      return;
    }
    // 如果不是叶子结点，在分别遍历他的左右子结点
    dfs13(root.left, path + root.val + "->", res);
    dfs13(root.right, path + root.val + "->", res);
  }
}

/** 二叉搜索树，中序为主 */
class BBST {
  private int count, res4;
  // 当前和上次遍历结点，按照中序遍历，后者即前者的左侧
  private TreeNode pre, cur;

  /**
   * 二叉搜索树中的第k大的元素，右中左，对 k 做减法
   *
   * <p>扩展1，二叉搜索树中的第k小的元素，左中右，参下 annotate
   *
   * @param root the root
   * @param k the k
   * @return int int
   */
  public int kthSmallest(TreeNode root, int k) {
    int count = k;
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.push(cur);
        cur = cur.right; // left
      }
      cur = stack.pop();
      count -= 1;
      if (count == 0) return cur.val;
      cur = cur.left; // right
    }
    // impossible
    return -1;
  }

  /**
   * 二叉搜索树中的插入操作，比当前小，则入左，否则入右
   *
   * @param root the root
   * @param val the val
   * @return tree node
   */
  public TreeNode insertIntoBST(TreeNode root, int val) {
    if (root == null) return new TreeNode(val);
    TreeNode cur = root;
    while (true) {
      if (cur.val < val) {
        if (cur.right == null) {
          cur.right = new TreeNode(val);
          return root;
        } else {
          cur = cur.right;
        }
      } else {
        if (cur.left == null) {
          cur.left = new TreeNode(val);
          return root;
        } else {
          cur = cur.left;
        }
      }
    }
  }

  /**
   * 删除二叉搜索树中的结点，递归找 target & 右子最左接 target 左 & 右子上位
   *
   * @param root the root
   * @param key the key
   * @return tree node
   */
  public TreeNode deleteNode(TreeNode root, int key) {
    if (root == null) return null;
    if (key < root.val) root.left = deleteNode(root.left, key);
    else if (key > root.val) root.right = deleteNode(root.right, key);
    else {
      if (root.left == null) return root.right;
      else if (root.right == null) return root.left;
      // 左右均非空
      TreeNode cur = root.right;
      while (cur.left != null) {
        cur = cur.left;
      }
      cur.left = root.left;
      // 右子上位
      root = root.right;
    }
    return root;
  }

  /**
   * 二叉树搜索的最小绝对差，中序，框架等同上方 inorderTraversal
   *
   * @param root the root
   * @return minimum difference
   */
  public int getMinimumDifference(TreeNode root) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    int res = Integer.MAX_VALUE;
    int pre = res;
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.addLast(cur);
        cur = cur.left;
      }
      cur = stack.removeLast();
      res = Math.min(res, Math.abs(cur.val - pre));
      pre = cur.val;
      cur = cur.right;
    }
    return res;
  }

  /**
   * 恢复二叉搜索树，中序，框架保持 inorderTraversal
   *
   * <p>找到两个错误结点并交换
   *
   * @param root the root
   */
  public void recoverTree(TreeNode root) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode firstNode = null, secondNode = null;
    TreeNode pre = new TreeNode(Integer.MIN_VALUE), cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.addLast(cur);
        cur = cur.left;
      }
      cur = stack.removeLast();
      if (firstNode == null && pre.val > cur.val) firstNode = pre;
      if (firstNode != null && pre.val > cur.val) secondNode = cur;
      pre = cur;
      cur = cur.right;
    }
    int tmp = firstNode.val;
    firstNode.val = secondNode.val;
    secondNode.val = tmp;
  }

  /**
   * 将有序数组转换为二叉搜索树，前序，类似双路快排，以升序数组的中间元素作 root
   *
   * @param nums the nums
   * @return tree node
   */
  public TreeNode sortedArrayToBST(int[] nums) {
    return dfs6(nums, 0, nums.length - 1);
  }

  private TreeNode dfs6(int[] nums, int lo, int hi) {
    if (lo > hi) return null;
    int mid = lo + (hi - lo) / 2;
    TreeNode root = new TreeNode(nums[mid]);
    root.left = dfs6(nums, lo, mid - 1);
    root.right = dfs6(nums, mid + 1, hi);
    return root;
  }

  /**
   * 二叉搜索树与双向链表，生成正序链表，中序
   *
   * <p>扩展1，逆序
   *
   * @param root the root
   * @return tree node
   */
  public TreeNode treeToDoublyList(TreeNode root) {
    if (root == null) return null;
    dfs7(root);
    // 关联头尾
    pre.right = cur;
    cur.left = pre;
    return cur;
  }

  // 每次只处理左侧，即前驱
  private void dfs7(TreeNode head) {
    if (head == null) return;
    dfs7(head.left);
    // 左
    if (pre == null) cur = head;
    else pre.right = head;
    // 右
    head.left = pre;
    pre = head;
    dfs7(head.right);
  }

  /**
   * 二叉搜索树迭代器
   *
   * <p>参考
   * https://leetcode-cn.com/problems/binary-search-tree-iterator/solution/fu-xue-ming-zhu-dan-diao-zhan-die-dai-la-dkrm/
   *
   * <p>均摊复杂度是 O(1)，调用 next()，如果栈顶元素有右子树，则把所有右边结点即其所有左孩子全部放到了栈中，下次调用 next() 直接访问栈顶
   *
   * <p>空间复杂度 O(h)，h 为数的高度，因为栈中只保留了左结点，栈中元素最多时，即树高
   */
  public class BSTIterator {
    private final Deque<TreeNode> stack = new ArrayDeque<>();

    /**
     * Instantiates a new Bst iterator.
     *
     * @param root the root
     */
    public BSTIterator(TreeNode root) {
      TreeNode cur = root;
      while (cur != null) {
        stack.push(cur);
        cur = cur.left;
      }
    }

    /**
     * Next int.
     *
     * @return the int
     */
    public int next() {
      TreeNode cur = stack.pop();
      if (cur.right != null) {
        TreeNode nxt = cur.right;
        while (nxt != null) {
          stack.push(nxt);
          nxt = nxt.left;
        }
      }
      return cur.val;
    }

    /**
     * Has next boolean.
     *
     * @return the boolean
     */
    public boolean hasNext() {
      return !stack.isEmpty();
    }
  }
}

/** 后序相关，常见为统计，自顶向下的递归相当于前序遍历，自底向上的递归相当于后序遍历 */
class Postorder {
  private int res1 = Integer.MIN_VALUE;
  private int res2 = 0;
  private String path;

  /**
   * 平衡二叉树，前序递归
   *
   * @param root the root
   * @return boolean
   */
  public boolean isBalanced(TreeNode root) {
    return getHeight(root) != -1;
  }

  private int getHeight(TreeNode root) {
    if (root == null) return 0;
    int left = getHeight(root.left);
    if (left == -1) return -1;
    int right = getHeight(root.right);
    if (right == -1) return -1;
    return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
  }

  /**
   * 二叉树的最近公共祖先，后序遍历
   *
   * <p>特判 & 剪枝，判断 p & q 均在左子树内 & 返回非空结点
   *
   * <p>TODO 扩展1，n 叉树，先找到两条路径查找第一个相交的结点即可，如果用后序遍历需要把每个分支都写清楚是否为空比较麻烦
   *
   * @param root the root
   * @param p the p
   * @param q the q
   * @return tree node
   */
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) {
      return root;
    }
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    if (left != null && left != q && left != p) {
      return left;
    }
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    if (left != null && right != null) {
      return root;
    }
    return left == null ? right : left;
  }

  /**
   * 二叉搜索树展开为链表，后序遍历
   *
   * @param root the root
   */
  public void flatten(TreeNode root) {
    if (root == null) return;
    flatten(root.left);
    flatten(root.right);
    TreeNode oldRight = root.right;
    root.right = root.left;
    root.left = null;
    TreeNode tailRight = root;
    while (tailRight.right != null) tailRight = tailRight.right;
    tailRight.right = oldRight;
  }

  /**
   * 二叉树中的最大路径和，后序遍历，模板与「二叉树但直径」近乎一致
   *
   * <p>三步曲，先取单侧 & 更新双侧结果 & 返回单侧更大者
   *
   * <p>扩展1，输出路径，参下 dfs
   *
   * @param root the root
   * @return int int
   */
  public int maxPathSum(TreeNode root) {
    singleSide1(root);
    return res1;
  }

  private int singleSide1(TreeNode root) {
    if (root == null) return 0;
    int left = Math.max(0, singleSide1(root.left)), right = Math.max(0, singleSide1(root.right));
    res1 = Math.max(res1, left + right + root.val);
    return Math.max(left, right) + root.val;
  }

  // https://blog.csdn.net/Ackerman2/article/details/119060128
  private Res _singleSide1(TreeNode root) {
    if (root == null) return new Res(0);
    Res res = new Res(root.val);
    Res left = _singleSide1(root.left), right = _singleSide1(root.right);
    if (left.count > 0 && left.count > right.count) {
      res.count += left.count;
      res.path += left.path;
    } else if (right.count > 0 && right.count > left.count) {
      res.count += right.count;
      res.path += right.path;
    }
    if (res.count > res1) {
      res1 = res.count;
      path = res.path;
    }
    return res;
  }

  /**
   * 二叉树的直径，后序遍历，模板与「二叉树的最大路径和」近乎一致
   *
   * @param root the root
   * @return int int
   */
  public int diameterOfBinaryTree(TreeNode root) {
    singleSide2(root);
    return res2 - 1;
  }

  private int singleSide2(TreeNode root) {
    if (root == null) return 0;
    int left = Math.max(0, singleSide2(root.left)), right = Math.max(0, singleSide2(root.right));
    res2 = Math.max(res2, left + right + 1);
    return Math.max(left, right) + 1;
  }

  /**
   * 不同的二叉搜索树II，后序遍历
   *
   * <p>参考
   * https://leetcode-cn.com/problems/unique-binary-search-trees-ii/solution/cong-gou-jian-dan-ke-shu-dao-gou-jian-suo-you-shu-/
   *
   * @param n the n
   * @return list
   */
  public List<TreeNode> generateTrees(int n) {
    return n < 1 ? new ArrayList<>() : dfs10(1, n);
  }

  private List<TreeNode> dfs10(int start, int end) {
    List<TreeNode> res = new ArrayList<>();
    if (start > end) {
      res.add(null);
      return res;
    }
    for (int i = start; i <= end; i++) {
      List<TreeNode> leftList = dfs10(start, i - 1), rightList = dfs10(i + 1, end);
      // 固定左，遍历右
      for (TreeNode left : leftList) {
        for (TreeNode right : rightList) {
          TreeNode root = new TreeNode(i);
          root.left = left;
          root.right = right;
          res.add(root);
        }
      }
    }
    return res;
  }

  /**
   * 二叉树的最大深度，后序
   *
   * <p>扩展1，n 叉树，改用 bfs
   *
   * @param root the root
   * @return int int
   */
  public int maxDepth(TreeNode root) {
    return maxDepth2(root);
  }

  private int maxDepth1(TreeNode root) {
    if (root == null) return 0;
    int left = maxDepth1(root.left), right = maxDepth1(root.right);
    return left > right ? left + 1 : right + 1;
  }

  private int maxDepth2(TreeNode root) {
    if (root == null) {
      return 0;
    }
    Deque<TreeNode> stack = new ArrayDeque<>();
    // 下方先更新 depth 再赋值，因此 root 高度初始化此处可略
    TreeNode cur = root;
    int res = 0, depth = 0;
    while (!stack.isEmpty() || cur != null) {
      while (cur != null) {
        depth += 1;
        cur.val = depth;
        stack.push(cur);
        cur = cur.left;
      }
      // 若左边无路，就预备右拐。右拐之前，记录右拐点的基本信息
      cur = stack.removeLast();
      // 将右拐点出栈；此时栈顶为右拐点的前一个结点。在右拐点的右子树全被遍历完后，会预备在这个结点右拐
      depth = cur.val;
      // 预备右拐时，比较当前结点深度和之前存储的最大深度
      res = Math.max(res, depth);
      cur = cur.right;
    }
    return res;
  }

  private class Res {
    /** The Count. */
    int count;

    /** The Path. */
    String path;

    /**
     * Instantiates a new Res.
     *
     * @param count the count
     */
    public Res(int count) {
      this.count = count;
      this.path = String.valueOf(count);
    }
  }
}

/** 广度优先搜索 */
class BBFS {
  /**
   * 二叉树的层序遍历，递归实现，前序，记录 level 即可
   *
   * @param root the root
   * @return list
   */
  public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    if (root != null) dfs8(res, root, 0);
    return res;
  }

  private void dfs8(List<List<Integer>> res, TreeNode node, int level) {
    // 需要加层
    if (res.size() - 1 < level) res.add(new ArrayList<Integer>());
    res.get(level).add(node.val);
    if (node.left != null) dfs8(res, node.left, level + 1);
    if (node.right != null) dfs8(res, node.right, level + 1);
  }

  /**
   * 二叉树的右视图
   *
   * @param root the root
   * @return list list
   */
  public List<Integer> rightSideView(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    if (root == null) return res;
    Deque<TreeNode> queue = new LinkedList<>();
    queue.addLast(root);
    while (!queue.isEmpty()) {
      res.add(queue.getLast().val);
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.removeFirst();
        if (cur.left != null) queue.addLast(cur.left);
        if (cur.right != null) queue.addLast(cur.right);
      }
    }
    return res;
  }

  /**
   * 二叉树的锯齿形层序遍历
   *
   * @param root the root
   * @return list list
   */
  public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();
    if (root != null) {
      queue.add(root);
    }
    boolean isOdd = true;
    while (!queue.isEmpty()) {
      Deque<Integer> levelList = new LinkedList<>();
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        if (isOdd) {
          levelList.addLast(cur.val);
        } else {
          levelList.addFirst(cur.val);
        }
        if (cur.left != null) {
          queue.add(cur.left);
        }
        if (cur.right != null) {
          queue.add(cur.right);
        }
      }
      res.add(new ArrayList<>(levelList));
      isOdd = !isOdd;
    }
    return res;
  }

  /**
   * 二叉树的完全性校验
   *
   * @param root the root
   * @return boolean boolean
   */
  private boolean isCompleteTree(TreeNode root) {
    if (root == null) return true;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    boolean isPreNull = false;
    while (!queue.isEmpty()) {
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.remove();
        if (cur == null) {
          isPreNull = true;
          continue;
        }
        if (isPreNull) return false;
        queue.add(cur.left);
        queue.add(cur.right);
      }
    }
    return true;
  }

  /**
   * 二叉树最大宽度，原地修改，否则引入 hash 以 val 为 key 记录
   *
   * @param root the root
   * @return int int
   */
  public int widthOfBinaryTree(TreeNode root) {
    if (root == null) return 0;
    int res = 0;
    Deque<TreeNode> queue = new LinkedList<>();
    root.val = 0;
    queue.add(root);
    while (!queue.isEmpty()) {
      res = Math.max(res, queue.getLast().val - queue.getFirst().val + 1);
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.removeFirst();
        if (cur.left != null) {
          queue.add(cur.left);
          cur.left.val = cur.val * 2 + 1;
        }
        if (cur.right != null) {
          queue.add(cur.right);
          cur.right.val = cur.val * 2 + 2;
        }
      }
    }
    return res;
  }

  /**
   * 翻转二叉树，迭代 bfs，逐一交换遍历的结点的左右子树即可
   *
   * @param root the root
   * @return tree node
   */
  public TreeNode invertTree(TreeNode root) {
    if (root == null) return null;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      TreeNode cur = queue.poll();
      swap(cur);
      if (cur.left != null) queue.offer(cur.left);
      if (cur.right != null) queue.offer(cur.right);
    }
    return root;
  }

  private void swap(TreeNode cur) {
    TreeNode tmp = cur.left;
    cur.left = cur.right;
    cur.right = tmp;
  }
}

/**
 * 收集图相关，题型有如下类型
 *
 * <p>判断连通性，拓扑排序
 *
 * <p>求最短路径，Floyd 邻接矩阵 or Dijkstra 邻接表
 *
 * <p>求路径总数，Floyd
 */
class GGraph {
  /**
   * 课程表，判断连通性，拓扑排序
   *
   * <p>相当于 BFS，思想是贪心，每一次都从图中删除没有前驱，即入度为 0 的点，最终图中还有点没有被移除，说明图无环
   *
   * <p>并不需要真正删除，而设置一个入度数组，每一轮都输出入度为 0 的点，并移除它、修改它指向的结点的入度 -1 即可，依次得到的结点序列就是拓扑排序的结点序列
   *
   * <p>因此，至少需要入度数组 & 图 & 遍历队列三种数据结构，步骤如下
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
    // [1,0] 即 0->1
    for (int[] cp : prerequisites) {
      int fromID = cp[0], toID = cp[1];
      indegrees[fromID] += 1;
      graph.get(toID).add(fromID);
    }
    for (int i = 0; i < numCourses; i++) {
      if (indegrees[i] == 0) queue.add(i);
    }
    // BFS TopSort.
    while (!queue.isEmpty()) {
      int pre = queue.poll();
      // handle it
      numCourses -= 1;
      // traverse its adjacency
      for (int cur : graph.get(pre)) {
        indegrees[cur] -= 1;
        if (indegrees[cur] == 0) queue.add(cur);
      }
    }
    return numCourses == 0;
  }

  /**
   * 课程表II，为上方新增 res 记录即可
   *
   * <p>检测循环依赖同理，参考 https://mp.weixin.qq.com/s/pCRscwKqQdYYN7M1Sia7xA
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return int [ ]
   */
  public int[] findOrder(int numCourses, int[][] prerequisites) {
    int[] res = new int[numCourses];
    int[] indegrees = new int[numCourses];
    List<List<Integer>> graph = new ArrayList<>(numCourses);
    for (int i = 0; i < numCourses; i++) {
      graph.add(new ArrayList<>());
    }
    Queue<Integer> queue = new LinkedList<>();
    for (int[] cp : prerequisites) {
      int fromID = cp[0], toID = cp[1];
      indegrees[fromID] += 1;
      graph.get(toID).add(fromID);
    }
    for (int i = 0; i < numCourses; i++) {
      if (indegrees[i] == 0) queue.add(i);
    }
    // 当前结果集的元素个数，正好可作为下标
    int count = 0;
    while (!queue.isEmpty()) {
      int pre = queue.poll();
      res[count] = pre;
      count += 1;
      for (int cur : graph.get(pre)) {
        indegrees[cur] -= 1;
        if (indegrees[cur] == 0) queue.add(cur);
      }
    }
    // 如果结果集中的数量不等于结点的数量，就不能完成课程任务，这一点是拓扑排序的结论
    return count == numCourses ? res : new int[0];
  }
}
