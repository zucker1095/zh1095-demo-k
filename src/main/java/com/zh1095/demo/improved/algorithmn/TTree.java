package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集树相关，BFS 参下，其余包括
 *
 * <p>关于 Java 模拟 stack 的选型
 * https://qastack.cn/programming/6163166/why-is-arraydeque-better-than-linkedlist
 *
 * <p>前序，本类
 *
 * <p>中序，基本即是 BST
 *
 * <p>后序，统计相关，参下
 *
 * @author cenghui
 */
public class TTree {
  /**
   * 中序遍历迭代，注意区分遍历 & 处理两个 step
   *
   * <p>https://leetcode-cn.com/problems/binary-tree-postorder-traversal/solution/zhuan-ti-jiang-jie-er-cha-shu-qian-zhong-hou-xu-2/
   *
   * <p>通过 Deque 模拟 stack
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
        // res = append(res, cur.Val) // pre & post
        stack.addLast(cur); // 1.traverse
        cur = cur.left; // right pre & left post
      }
      cur = stack.removeLast(); // 2.handle
      res.add(cur.val);
      cur = cur.right; // 3.下一跳 left pre & right post
    }
    // Collections.reverse(res); // post
    return res;
  }

  private void traverse(Deque<TreeNode> stack, TreeNode root) {
    stack.addLast(root);
  }

  private TreeNode handle(Deque<TreeNode> stack, List<Integer> res) {
    TreeNode last = stack.getLast();
    stack.removeLast();
    res.add(last.val);
    return last;
  }

  /**
   * 从前序与中序遍历序列构造二叉树
   *
   * @param preorder the preorder
   * @param inorder the inorder
   * @return the tree node
   */
  public TreeNode buildTree(int[] preorder, int[] inorder) {
    Map<Integer, Integer> idxByValInorder = new HashMap<Integer, Integer>();
    for (int i = 0; i < preorder.length; i++) idxByValInorder.put(inorder[i], i);
    return buildTree(preorder, 0, preorder.length - 1, idxByValInorder, 0);
  }

  private TreeNode buildTree(
      int[] preorder, int preLo, int preHi, Map<Integer, Integer> idxByValInorder, int inLo) {
    if (preLo > preHi) return null;
    TreeNode root = new TreeNode(preorder[preLo]);
    int idx = idxByValInorder.get(preorder[preLo]);
    int countLeft = idx - inLo;
    root.left = buildTree(preorder, preLo + 1, preLo + countLeft, idxByValInorder, inLo);
    root.right = buildTree(preorder, preLo + countLeft + 1, preHi, idxByValInorder, idx + 1);
    return root;
  }
  /**
   * 求根节点到叶子节点数字之和，前序
   *
   * @param root the root
   * @return int int
   */
  public int sumNumbers(TreeNode root) {
    preOrdering(root, 0);
    return res3;
  }

  private int res3 = 0;

  private void preOrdering(TreeNode root, int path) {
    if (root == null) return;
    // 题设不会越界
    int cur = path * 10 + root.val;
    if (root.left == null && root.right == null) {
      res3 += cur;
      return;
    }
    preOrdering(root.left, cur);
    preOrdering(root.right, cur);
  }

  /**
   * 路径总和I，前序遍历，II 参下回溯
   *
   * @param root the root
   * @param targetSum the target sum
   * @return boolean boolean
   */
  public boolean hasPathSum(TreeNode root, int targetSum) {
    if (root == null) return false;
    if (root.left == null && root.right == null) return targetSum - root.val == 0;
    return hasPathSum(root.left, targetSum - root.val)
        || hasPathSum(root.right, targetSum - root.val); // 其一
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
   * 相同的树，前序
   *
   * @param p the p
   * @param q the q
   * @return boolean boolean
   */
  public boolean isSameTree(TreeNode p, TreeNode q) {
    if (p == null && q == null) return true;
    else if (p == null || q == null) return false;
    return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
  }

  /**
   * 另一棵树的子树
   *
   * <p>特判匹配树 & 主树为空两种情况
   *
   * <p>当然，isSameTree 中的两处特判可以去除，因为匹配树 & 主树均非空
   *
   * @param root the root
   * @param subRoot the sub root
   * @return boolean
   */
  public boolean isSubtree(TreeNode root, TreeNode subRoot) {
    if (subRoot == null) {
      return true;
    }
    if (root == null) {
      return false;
    }
    return isSubtree(root.left, subRoot)
        || isSubtree(root.right, subRoot)
        || isSameTree(root, subRoot);
  }
}

/** 后序相关，常见为统计 */
class Postorder {

  /**
   * 二叉树的最近公共祖先，后序遍历
   *
   * @param root the root
   * @param p the p
   * @param q the q
   * @return tree node
   */
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || p == root || q == root) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q),
        right = lowestCommonAncestor(root.right, p, q);
    return left == null ? right : (right == null ? left : root);
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
   * 二叉树中的最大路径和，后序遍历
   *
   * @param root the root
   * @return int int
   */
  public int maxPathSum(TreeNode root) {
    singleSide1(root);
    return res1;
  }

  private int res1 = Integer.MIN_VALUE;

  private int singleSide1(TreeNode root) {
    if (root == null) return 0;
    int left = Math.max(0, singleSide1(root.left)), right = Math.max(0, singleSide1(root.right));
    res1 = Math.max(res1, left + right + root.val);
    return Math.max(left, right) + root.val;
  }

  /**
   * 二叉树的直径，后序遍历
   *
   * @param root the root
   * @return int int
   */
  public int diameterOfBinaryTree(TreeNode root) {
    singleSide2(root);
    return res2 - 1;
  }

  private int res2 = 0;

  private int singleSide2(TreeNode root) {
    if (root == null) return 0;
    int left = Math.max(0, singleSide2(root.left)), right = Math.max(0, singleSide2(root.right));
    res2 = Math.max(res2, left + right + 1);
    return Math.max(left, right) + 1;
  }

  /**
   * 二叉树的最大深度，后序遍历
   *
   * @param root the root
   * @return int int
   */
  public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    int left = maxDepth(root.left), right = maxDepth(root.right);
    return left > right ? left + 1 : right + 1;
  }
}

/** 二叉搜索树，中序为主 */
class BBST {

  /**
   * 二叉搜索树中的第k小的元素，对 k 做减法，第 k 大则 right & root & left 做中序，参下
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
        cur = cur.left;
      }
      cur = stack.pop();
      count -= 1;
      if (count == 0) return cur.val;
      cur = cur.right;
    }
    return -1; // never
  }

  private int kthSmallest2(TreeNode root, int k) {
    count = k;
    inordering(root);
    return res4;
  }

  private int count, res4;

  private void inordering(TreeNode root) {
    if (root == null || count <= 0) return;
    inordering(root.left); // right
    count -= 1;
    if (count == 0) res4 = root.val;
    inordering(root.right); // left
  }

  /**
   * 二叉搜索树中的第k大的元素
   *
   * @param root the root
   * @param k the k
   * @return the int
   */
  public int kthLargest(TreeNode root, int k) {
    int count = k;
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.push(cur);
        cur = cur.right;
      }
      cur = stack.pop();
      count -= 1;
      if (count == 0) return cur.val;
      cur = cur.left;
    }
    return -1; // never
  }

  /**
   * 删除二叉搜索树中的节点
   *
   * <p>1.寻找 target
   *
   * <p>2.寻找其右子树的最左结点 cur 并赋值 target
   *
   * <p>3.删除 cur
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
      // target 左右结点分三种情况
      if (root.left == null) return root.right;
      else if (root.right == null) return root.left;
      TreeNode pre = root, cur = root.right;
      while (cur.left != null) {
        pre = cur;
        cur = cur.left;
      }
      root.val = cur.val;
      // 将下一个值的节点删除
      if (pre.left.val == cur.val) pre.left = cur.right;
      else pre.right = cur.right;
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
   * 将有序数组转换为二叉搜索树，以升序数组的中间元素作 root
   *
   * @param nums the nums
   * @return tree node
   */
  public TreeNode sortedArrayToBST(int[] nums) {
    return dfs(nums, 0, nums.length - 1);
  }

  private TreeNode dfs(int[] nums, int lo, int hi) {
    if (lo > hi) return null;
    int mid = lo + (hi - lo) / 2;
    TreeNode root = new TreeNode(nums[mid]);
    root.left = dfs(nums, lo, mid - 1);
    root.right = dfs(nums, mid + 1, hi);
    return root;
  }
}

/** BFS */
class BBFS {
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
    if (root == null) return res;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    boolean isOdd = true;
    while (!queue.isEmpty()) {
      Deque<Integer> levelList = new LinkedList<Integer>();
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        if (isOdd) levelList.addLast(cur.val);
        else levelList.addFirst(cur.val);
        if (cur.left != null) queue.add(cur.left);
        if (cur.right != null) queue.add(cur.right);
      }
      res.add(new LinkedList<>(levelList));
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
  public boolean isCompleteTree(TreeNode root) {
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
}
/**
 * 回溯，前序与后序结合，遵从如下规范
 *
 * <p>入参顺序为 nums, path, res(if need), ...args
 *
 * <p>按照子组列的顺序，建议按照表格记忆
 */
class BBacktracking extends DDFS {
  /**
   * 路径总和II，需要记录路径
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
      res.add(new ArrayList<>(path)); // path 全局只有一份，必须做拷贝
      path.removeLast(); // return 之前必须重置
      return;
    }
    backtracking0(root.left, path, res, targetSum - root.val);
    backtracking0(root.right, path, res, targetSum - root.val);
    // 递归完成以后，必须重置变量
    path.removeLast();
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
   * 复原IP地址
   *
   * @param s the s
   * @return list list
   */
  public List<String> restoreIpAddresses(String s) {
    List<String> res = new ArrayList<>();
    if (s.length() > 12 || s.length() < 4) return res; // 特判
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
      if (residue * 3 < s.length() - i) continue;
      if (isValidIpSegment(s, start, i)) {
        path.addLast(s.substring(start, i + 1));
        backtracking6(s, path, res, i + 1, residue - 1);
        path.removeLast();
      }
    }
  }

  private boolean isValidIpSegment(String s, int lo, int hi) {
    int res = 0;
    if (hi > lo && s.charAt(lo) == '0') return false;
    while (lo <= hi) {
      res = res * 10 + s.charAt(lo) - '0';
      lo += 1;
    }
    return res >= 0 && res <= 255;
  }

  /**
   * 括号生成
   *
   * @param n the n
   * @return list list
   */
  public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    if (n <= 0) return res; // 特判
    backtracking7(n, n, "", res);
    return res;
  }
  // 此处的可选集为左右括号的剩余量
  // 因为每一次尝试，都使用新的字符串变量，所以无需显示回溯
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
    if (board.length == 0) return false;
    int rows = board.length, cols = board[0].length;
    boolean[][] visited = new boolean[rows][cols];
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) if (backtracking8(board, i, j, word, 0, visited)) return true;
    return false;
  }

  private boolean backtracking8(
      char[][] board, int r, int c, String word, int start, boolean[][] visited) {
    if (start == word.length() - 1) return board[r][c] == word.charAt(start);
    if (board[r][c] != word.charAt(start)) return false;
    visited[r][c] = true;
    for (int[] dir : DIRECTIONS) {
      int newX = r + dir[0];
      int newY = c + dir[1];
      if (inArea(board, newX, newY)
          && !visited[newX][newY]
          && backtracking8(board, newX, newY, word, start + 1, visited)) return true;
    }
    visited[r][c] = false;
    return false;
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
  public int _pathSum(TreeNode root, int targetSum) {
    Map<Long, Integer> prefix =
        new HashMap<>() {
          {
            put(0L, 1); // base case
          }
        };
    return backtracking9(root, prefix, 0, targetSum);
  }

  private int backtracking9(TreeNode root, Map<Long, Integer> prefix, long cur, int targetSum) {
    if (root == null) return 0;
    cur += root.val;
    int res = prefix.getOrDefault(cur - targetSum, 0);
    prefix.put(cur, prefix.getOrDefault(cur, 0) + 1);
    res +=
        backtracking9(root.left, prefix, cur, targetSum)
            + backtracking9(root.right, prefix, cur, targetSum);
    prefix.put(cur, prefix.getOrDefault(cur, 0) - 1);
    return res;
  }
}

/**
 * 深度优先搜索
 *
 * <p>回溯和 dfs 框架基本一致，但前者适用 tree 这类不同分支互不连通的结构，而后者更适合 graph 这类各个分支都可能连通的
 *
 * <p>因此后者不需要回溯，比如下方 grid[i][j]=2 后不需要再恢复，因为要避免环路
 */
class DDFS {
  /**
   * 岛屿数量
   *
   * <p>https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/
   *
   * <p>扩展，假如岛屿有权重，要求路径递增
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
    if (inArea(grid, r, c) && grid[r][c] != '0') {
      grid[r][c] = '0';
      for (int[] dir : DIRECTIONS) dfs1(grid, r + dir[0], c + dir[1]);
    }
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
    if (inArea(grid, r, c) && grid[r][c] != 0) {
      grid[r][c] = 0;
      int res = 1;
      for (int[] dir : DIRECTIONS) res += dfs2(grid, r + dir[0], c + dir[1]);
      return res;
    }
    return 0;
  }

  /**
   * 矩阵中的最长递增路径
   *
   * @param matrix the matrix
   * @return int int
   */
  public int longestIncreasingPath(int[][] matrix) {
    if (matrix == null || matrix.length == 0 || matrix[0].length == 0) return 0;
    int rows = matrix.length, columns = matrix[0].length;
    int[][] memo = new int[rows][columns];
    int res = 0;
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < columns; j++) res = Math.max(res, dfs3(matrix, i, j, memo));
    return res;
  }

  private int dfs3(int[][] matrix, int r, int c, int[][] memo) {
    if (memo[r][c] != 0) return memo[r][c];
    memo[r][c] += 1;
    for (int[] dir : DIRECTIONS) {
      int newR = r + dir[0], newC = c + dir[1];
      if (inArea(matrix, newR, newC) && matrix[newR][newC] <= matrix[r][c])
        memo[r][c] = Math.max(memo[r][c], dfs3(matrix, newR, newC, memo) + 1);
    }
    return memo[r][c];
  }

  /** The Directions. */
  protected final int[][] DIRECTIONS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  /**
   * In area boolean.
   *
   * @param board the board
   * @param i the
   * @param j the j
   * @return the boolean
   */
  protected boolean inArea(int[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }

  /**
   * In area boolean.
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
   * 二叉树中所有距离为k的结点
   *
   * <p>建图，从 root 出发 DFS 记录每个结点的父结点
   *
   * <p>从 target 出发 DFS 寻找相距为 k 的点
   *
   * @param root the root
   * @param target the target
   * @param k the k
   * @return list list
   */
  public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
    List<Integer> res = new ArrayList<>();
    Map<Integer, TreeNode> parents = new HashMap<>();
    findParents(root, parents);
    dfs4(parents, res, target, null, k);
    return res;
  }

  private void findParents(TreeNode node, Map<Integer, TreeNode> parents) {
    if (node.left != null) {
      parents.put(node.left.val, node);
      findParents(node.left, parents);
    }
    if (node.right != null) {
      parents.put(node.right.val, node);
      findParents(node.right, parents);
    }
  }

  private void dfs4(
      Map<Integer, TreeNode> parents, List<Integer> res, TreeNode node, TreeNode from, int left) {
    if (node == null) return;
    if (left == 0) {
      res.add(node.val);
      return;
    }
    if (node.left != from) dfs4(parents, res, node.left, node, left - 1);
    if (node.right != from) dfs4(parents, res, node.right, node, left - 1);
    if (parents.get(node.val) != from) dfs4(parents, res, parents.get(node.val), node, left - 1);
  }
}

/** 收集图相关 */
class GGraph {
  /**
   * 课程表，判断连通性，拓扑排序
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return boolean boolean
   */
  public boolean canFinish(int numCourses, int[][] prerequisites) {
    int[] flags = new int[numCourses];
    List<List<Integer>> adjacency = new ArrayList<>();
    for (int i = 0; i < numCourses; i++) {
      adjacency.add(new ArrayList<>());
    }
    for (int[] cp : prerequisites) {
      adjacency.get(cp[1]).add(cp[0]);
    }
    for (int i = 0; i < numCourses; i++) {
      if (!dfs(adjacency, flags, i)) return false;
    }
    return true;
  }

  private boolean dfs(List<List<Integer>> adjacency, int[] flags, int i) {
    if (flags[i] == 1) {
      return false;
    } else if (flags[i] == -1) {
      return true;
    }
    flags[i] = 1;
    for (int j : adjacency.get(i)) {
      if (!dfs(adjacency, flags, j)) return false;
    }
    flags[i] = -1;
    return true;
  }

  /**
   * 课程表II，拓扑排序
   *
   * <p>DFS遍历，在选择下一步的结点的时候只能选择入度为 0 即更新后的结点
   *
   * <p>可能最外层循环中会重复出现入度为 0 的结点，因此可在遍历了入度为 0 的结点之后将其入度修改为 -1 以避免重复遍历
   *
   * <p>扩展1，检测循环依赖，只检测
   *
   * @param numCourses the num courses
   * @param prerequisites the prerequisites
   * @return int [ ]
   */
  public int[] findOrder(int numCourses, int[][] prerequisites) {
    int[] inDegree = new int[numCourses];
    // 根据有向关系建图并记录每个结点的入度大小
    int[] visitOrder = new int[numCourses];
    Map<Integer, List<Integer>> graph = new HashMap<>(prerequisites.length);
    for (int i = 0; i < numCourses; i++) {
      for (int[] prerequisite : prerequisites) {
        int first = prerequisite[1], next = prerequisite[0];
        inDegree[next] += 1;
        if (!graph.containsKey(first)) graph.put(first, new ArrayList<>());
        graph.get(first).add(next);
      }
      if (order == numCourses) break;
      // 仅入度为 0 的结点可被立刻遍历
      if (inDegree[i] == 0) dfs(i, inDegree, visitOrder, graph);
    }
    return (order == numCourses) ? visitOrder : new int[0];
  }

  private int order;

  private void dfs(int pos, int[] inDegree, int[] visitOrder, Map<Integer, List<Integer>> graph) {
    visitOrder[order] = pos;
    order += 1;
    // 将入度为 0 的结点入度改为 -1 以剪枝
    inDegree[pos] -= 1;
    for (int next : graph.getOrDefault(pos, new ArrayList<>())) {
      inDegree[next] -= 1;
      if (inDegree[next] == 0) dfs(next, inDegree, visitOrder, graph);
    }
  }
}
