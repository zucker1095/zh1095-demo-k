package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集树相关，BFS 参下，其余包括
 *
 * <p>关于 Java 模拟 stack 的选型
 * https://qastack.cn/programming/6163166/why-is-arraydeque-better-than-linkedlist
 *
 * <p>前序尝试均改为迭代
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
  // 「求根节点到叶节点数字之和」
  private int res3 = 0;

  /**
   * 中序遍历，迭代，注意区分遍历 & 处理两个 step
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
        stack.offerLast(cur); // 1.traverse
        cur = cur.left;
      }
      cur = stack.pollLast(); // 2.handle
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
        stack.offerLast(cur);
        cur = cur.right; // left pre
      }
      cur = stack.pollLast();
      cur = cur.left; // right pre
    }
    Collections.reverse(res);
    return res;
  }

  /**
   * 构造二叉树，题设元素唯一，否则，存在多棵树
   *
   * <p>扩展1，根据前序和中序，输出后序，不能构造树，参下 annotate
   *
   * <p>扩展2，给一个随机数组，生成相应的二叉搜索树，先排序，参下「将有序数组转换为二叉搜索树」
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

  // 从前序与中序遍历序列构造二叉树
  private TreeNode buildTree1(
      int[] preorder, int preLo, int preHi, Map<Integer, Integer> idxByValInorder, int inLo) {
    //    if (preLo == preHi) {
    //      postorder.add(preorder[preLo]);
    //      return;
    //    }
    if (preLo > preHi) return null;
    TreeNode root = new TreeNode(preorder[preLo]);
    int idx = idxByValInorder.get(preorder[preLo]), countLeft = idx - inLo;
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
    int idx = idxByValInorder.get(postrorder[postHi]), countLeft = idx - inLo;
    root.left = buildTree2(postrorder, postLo, postLo + countLeft - 1, idxByValInorder, inLo);
    root.right = buildTree2(postrorder, postLo + countLeft, postHi - 1, idxByValInorder, idx + 1);
    return root;
  }

  /**
   * 路径总和，从根出发要求达到叶，BFS / 前序，前者模板与下方「求根节点到叶节点数字之和」一致
   *
   * <p>记录路径则参下「路径总和II」回溯
   *
   * @param root the root
   * @param targetSum the target sum
   * @return boolean boolean
   */
  public boolean hasPathSum(TreeNode root, int sum) {
    if (root == null) return false;
    Queue<TreeNode> nodeQueue = new LinkedList<>();
    Queue<Integer> numQueue = new LinkedList<>();
    nodeQueue.offer(root);
    numQueue.offer(root.val);
    while (!nodeQueue.isEmpty()) {
      TreeNode node = nodeQueue.poll();
      int num = numQueue.poll();
      TreeNode left = node.left, right = node.right;
      if (left == null && right == null) {
        if (num == sum) return true;
        continue;
      }
      if (left != null) {
        nodeQueue.offer(left);
        numQueue.offer(left.val + num);
      }
      if (right != null) {
        nodeQueue.offer(right);
        numQueue.offer(right.val + num);
      }
    }
    return false;
  }

  /**
   * 求根节点到叶节点数字之和，BFS 维护两个队列逐层相加 / 前序，前者参上「路径总和」
   *
   * @param root the root
   * @return int int
   */
  public int sumNumbers(TreeNode root) {
    //    dfs12(root, 0);
    //    return res3;
    if (root == null) return 0;
    // 题设不会越界
    int sum = 0;
    Queue<TreeNode> nodeQueue = new LinkedList<>();
    Queue<Integer> numQueue = new LinkedList<>();
    nodeQueue.offer(root);
    numQueue.offer(root.val);
    while (!nodeQueue.isEmpty()) {
      TreeNode cur = nodeQueue.poll();
      int num = numQueue.poll();
      TreeNode left = cur.left, right = cur.right;
      if (left == null && right == null) {
        sum += num;
        continue;
      }
      if (left != null) {
        nodeQueue.offer(left);
        numQueue.offer(left.val + num * 10);
      }
      if (right != null) {
        nodeQueue.offer(right);
        numQueue.offer(right.val + num * 10);
      }
    }
    return sum;
  }

  private void dfs12(TreeNode root, int num) {
    if (root == null) return;
    // 题设不会越界
    int cur = num * 10 + root.val;
    if (root.left == null && root.right == null) {
      res3 += cur;
      return;
    }
    dfs12(root.left, cur);
    dfs12(root.right, cur);
  }

  /**
   * 二叉树的下一个节点，给定一棵树中任一结点，返回其中序遍历顺序的下一个结点，提供结点的 next 指向父结点
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
      if (cur.next.left.equals(cur)) return cur.next;
      cur = cur.next;
    }
    // 退到根结点仍没找到
    return null;
  }

  private class _TreeNode {
    private _TreeNode left, right, next;
  }

  /**
   * 二叉树的序列化与反序列化，前序
   *
   * <p>扩展1，多叉树，记录子树个数，参考 https://zhuanlan.zhihu.com/p/109521420
   */
  public class Codec {
    private int idx;

    /**
     * Serialize string.
     *
     * @param root the root
     * @return the string
     */
    public String serialize(TreeNode root) {
      return root == null
          ? "null"
          : root.val + "," + serialize(root.left) + "," + serialize(root.right);
      //      if (root == null) return "null";
      //      StringBuilder str = new StringBuilder();
      //      str.append(root.val);
      //      str.append(',');
      //      str.append(root.chlidren.length);
      //      for (int i = 1; i < root.children.length; i++) {
      //        str.append(',');
      //        str.append(root.children[i].val);
      //        str.append(',');
      //        str.append(root.chlidren.length);
      //      }
      //      return str.toString();
    }

    /**
     * Deserialize tree node.
     *
     * @param data the data
     * @return the tree node
     */
    public TreeNode deserialize(String data) {
      return traversal(data.split(","));
    }

    private TreeNode traversal(String[] vals) {
      if (idx >= vals.length) return null;
      String val = vals[idx], count = vals[idx + 1];
      idx += 2;
      if ("null".equals(val)) return null;
      TreeNode root = new TreeNode(Integer.parseInt(val));
      //      root.children = new TreeNode[count];
      //      for (int i = 0; i < count; i++) {
      //        root.children[i] = traversal(vals);
      //      }
      root.left = traversal(vals);
      root.right = traversal(vals);
      return root;
    }
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
  // 「二叉树中所有距离为k的结点」结果集
  private final List<Integer> res5 = new ArrayList<>();
  // 「二叉树中所有距离为k的结点」目标结点的父
  private TreeNode parent;

  /**
   * 坐标界内
   *
   * @param board the board
   * @param i the
   * @param j the j
   * @return the boolean
   */
  protected boolean inArea(char[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }

  private boolean inArea(int[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }

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
   * 矩阵中的最长递增路径
   *
   * @param matrix the matrix
   * @return int int
   */
  public int longestIncreasingPath(int[][] matrix) {
    if (matrix.length == 0 || matrix[0].length == 0) return 0;
    int rows = matrix.length, cols = matrix[0].length;
    int[][] memo = new int[rows][cols];
    int maxLen = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        maxLen = Math.max(maxLen, dfs3(matrix, i, j, memo));
      }
    }
    return maxLen;
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
    // 1.把树分为两棵，分别以目标结点及其父结点为根
    dfs4(null, root, target);
    // 2.以目标结点为根的中树深度为 k 的结点
    collect(target, k);
    // 3.以目标结点父为根的树，、深度为 k-1 的结点
    collect(parent, k - 1);
    return res5;
  }

  // 分割结点
  private Boolean dfs4(TreeNode from, TreeNode to, TreeNode target) {
    if (to == null) return false;
    // 如果搜到了目标结点，那么它父就是新树的根
    if (to == target) {
      parent = from;
      return true;
    }
    if (dfs4(to, to.left, target)) {
      to.left = from;
      return true;
    }
    if (dfs4(to, to.right, target)) {
      to.right = from;
      return true;
    }
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
   * 不同岛屿的数量
   *
   * @param grid the grid
   * @return the int
   */
  public int numDistinctIslands(int[][] grid) {
    Set<String> path = new HashSet<>();
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] != 1) continue;
        StringBuilder curPath = new StringBuilder();
        dfs5(grid, curPath, i, j, i, j);
        path.add(curPath.toString());
      }
    }
    return path.size();
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
}

/** 二叉搜索树，中序为主，模板与「中序遍历」一致 */
class BBSTInorder {
  /**
   * 验证二叉搜索树，模板参上「中序遍历」
   *
   * @param root
   * @return
   */
  public boolean isValidBST(TreeNode root) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    // Integer.MIN_VALUE 即可，此处仅为通过官方示例
    double pre = -Double.MAX_VALUE;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.offerLast(cur);
        cur = cur.left;
      }
      cur = stack.pollLast();
      if (cur.val <= pre) return false;
      pre = cur.val;
      cur = cur.right;
    }
    return true;
  }

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
        stack.offerLast(cur);
        cur = cur.right;
      }
      cur = stack.pollLast();
      count -= 1;
      if (count == 0) return cur.val;
      cur = cur.left;
    }
    // impossible
    return -1;
  }

  /**
   * 二叉搜索树的最小绝对差，中序，框架等同上方「中序遍历」
   *
   * @param root the root
   * @return minimum difference
   */
  public int getMinimumDifference(TreeNode root) {
    int minDiff = Integer.MAX_VALUE, pre = minDiff;
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.offerLast(cur);
        cur = cur.left;
      }
      cur = stack.pollLast();
      minDiff = Math.min(minDiff, Math.abs(cur.val - pre));
      pre = cur.val;
      cur = cur.right;
    }
    return minDiff;
  }

  /**
   * 恢复二叉搜索树，中序，框架保持「中序遍历」
   *
   * <p>中序依次找两个错误结点 & 交换值
   *
   * @param root the root
   */
  public void recoverTree(TreeNode root) {
    TreeNode firstNode = null, secondNode = null, pre = new TreeNode(Integer.MIN_VALUE), cur = root;
    Deque<TreeNode> stack = new ArrayDeque<>();
    while (cur != null || !stack.isEmpty()) {
      // 遍历
      while (cur != null) {
        stack.offerLast(cur);
        cur = cur.left;
      }
      cur = stack.pollLast();
      // 处理当前结点
      if (firstNode == null && pre.val > cur.val) firstNode = pre;
      if (firstNode != null && pre.val > cur.val) secondNode = cur;
      pre = cur;
      // 步进
      cur = cur.right;
    }
    int tmp = firstNode.val;
    firstNode.val = secondNode.val;
    secondNode.val = tmp;
  }

  /**
   * 二叉搜索树迭代器
   *
   * <p>均摊复杂度是 O(1)，调用 next()，如果栈顶元素有右子树，则把所有右边结点即其所有左孩子全部放到了栈中，下次调用 next() 直接访问栈顶
   *
   * <p>空间复杂度 O(h)，h 为数的高度，因为栈中只保留了左结点，栈中元素最多时，即树高
   *
   * <p>参考
   * https://leetcode-cn.com/problems/binary-search-tree-iterator/solution/fu-xue-ming-zhu-dan-diao-zhan-die-dai-la-dkrm/
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
        stack.offerLast(cur);
        cur = cur.left;
      }
    }

    /**
     * Next int.
     *
     * @return the int
     */
    public int next() {
      TreeNode cur = stack.pollLast(), nxt = cur.right;
      while (nxt != null) {
        stack.offerLast(nxt);
        nxt = nxt.left;
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

/** 二叉搜索树，深搜为主 */
class BBSTDFS {
  // 「二叉搜索树与双向链表」上次与当前遍历结点，按照中序遍历，前者即后者的左子树
  private TreeNode pre, cur;

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
    while (cur != null) {
      if (cur.val < val) {
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
   * 将有序数组转换为二叉搜索树 / 最小高度树，前序，类似双路快排，以升序数组的中间元素作 root
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
   * 有序链表转换二叉搜索树
   *
   * @param head
   * @return
   */
  public TreeNode sortedListToBST(ListNode head) {
    if (head == null) return null;
    if (head.next == null) return new TreeNode(head.val);
    ListNode pre = null, lo = head, hi = head;
    while (hi != null && hi.next != null) {
      pre = lo;
      lo = lo.next;
      hi = hi.next.next;
    }
    pre.next = null;
    TreeNode root = new TreeNode(lo.val);
    root.left = sortedListToBST(head);
    root.right = sortedListToBST(lo.next);
    return root;
  }

  /**
   * 二叉搜索树与双向链表，生成正序链表，中序
   *
   * <p>扩展1，逆序，则右中左
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
  private void dfs7(TreeNode root) {
    if (root == null) return;
    dfs7(root.left);
    if (pre != null) pre.right = root; // 尾插
    else cur = root; // 最左叶
    root.left = pre; // 补充前驱
    pre = root;
    dfs7(root.right);
  }
}

/** 后序相关，常见为统计，自顶向下的递归相当于前序遍历，自底向上的递归相当于后序遍历 */
class Postorder {
  // 「二叉树的最大路径和」
  private int maxSum = Integer.MIN_VALUE;
  // 「二叉树中的最大路径和」follow up 需要输出的路径
  private String maxPath;
  // 「二叉树的直径」
  private int curDiameter = 0;

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
   * 二叉树展开为链表，后序遍历
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
    while (tailRight.right != null) {
      tailRight = tailRight.right;
    }
    tailRight.right = oldRight;
  }

  /**
   * 二叉树中的最大路径和，从任意结点出发，后序遍历，模板与「二叉树但直径」近乎一致
   *
   * <p>三步曲，先取单侧 & 更新双侧结果 & 返回单侧更大者
   *
   * <p>扩展1，输出路径，参考 https://blog.csdn.net/Ackerman2/article/details/119060128
   *
   * @param root the root
   * @return int int
   */
  public int maxPathSum(TreeNode root) {
    singleSide1(root);
    return maxSum;
  }

  private int singleSide1(TreeNode root) {
    if (root == null) return 0;
    int left = Math.max(0, singleSide1(root.left)), right = Math.max(0, singleSide1(root.right));
    // 更新双侧
    maxSum = Math.max(maxSum, left + right + root.val);
    // 返回单侧
    return Math.max(left, right) + root.val;
  }

  private Res _singleSide1(TreeNode root) {
    if (root == null) return new Res();
    Res cur = new Res(), left = _singleSide1(root.left), right = _singleSide1(root.right);
    if (left.count <= 0) {
      left.count = 0;
      left.path = "";
    } else {
      left.path = left.count + "->";
    }
    if (right.count <= 0) {
      right.count = 0;
      left.path = "";
    } else {
      right.path = "->" + right.count;
    }
    // 更新双侧
    if (root.val + left.count + right.count > maxSum) {
      maxSum = left.count + root.val + right.count;
      maxPath = left.path + root.val + right.path;
    }
    // 返回单侧
    if (left.count > right.count) {
      cur.count = left.count + root.val;
      cur.path = left.path + root.val;
    } else {
      cur.count = root.val + right.count;
      cur.path = root.val + right.path;
    }
    return cur;
  }

  /**
   * 二叉树的直径，后序遍历，模板与「二叉树的最大路径和」近乎一致
   *
   * @param root the root
   * @return int int
   */
  public int diameterOfBinaryTree(TreeNode root) {
    singleSide2(root);
    return curDiameter - 1;
  }

  private int singleSide2(TreeNode root) {
    if (root == null) return 0;
    int left = Math.max(0, singleSide2(root.left)), right = Math.max(0, singleSide2(root.right));
    curDiameter = Math.max(curDiameter, left + right + 1);
    return Math.max(left, right) + 1;
  }

  /**
   * 不同的二叉搜索树II，后序遍历，固定左遍历右
   *
   * <p>只返回总数参考「不同的二叉搜索树」
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

  private List<TreeNode> dfs10(int lo, int hi) {
    List<TreeNode> path = new ArrayList<>();
    if (lo > hi) {
      path.add(null);
      return path;
    }
    for (int i = lo; i <= hi; i++) {
      List<TreeNode> leftPath = dfs10(lo, i - 1), rightPath = dfs10(i + 1, hi);
      for (TreeNode left : leftPath) {
        for (TreeNode right : rightPath) {
          TreeNode root = new TreeNode(i);
          root.left = left;
          root.right = right;
          path.add(root);
        }
      }
    }
    return path;
  }

  private class Res {
    /** The Count. */
    int count;

    /** The Path. */
    String path;
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
   * 二叉树的层序遍历II，选用链表
   *
   * @param root
   * @return
   */
  public List<List<Integer>> levelOrderBottom(TreeNode root) {
    List<List<Integer>> res = new LinkedList<List<Integer>>();
    if (root == null) return res;
    Queue<TreeNode> queue = new LinkedList<TreeNode>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      List<Integer> curLevel = new ArrayList<Integer>();
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        curLevel.add(cur.val);
        TreeNode left = cur.left, right = cur.right;
        if (left != null) queue.offer(left);
        if (right != null) queue.offer(right);
      }
      // 因此上方选用链表
      res.add(0, curLevel);
    }
    return res;
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
    queue.offerLast(root);
    while (!queue.isEmpty()) {
      res.add(queue.peekLast().val);
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.pollFirst();
        if (cur.left != null) queue.offerLast(cur.left);
        if (cur.right != null) queue.offerLast(cur.right);
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
        if (cur.left != null) queue.offer(cur.left);
        if (cur.right != null) queue.offer(cur.right);

        if (isOdd) levelList.offerLast(cur.val);
        else levelList.offerFirst(cur.val);
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
    queue.offer(root);
    boolean isPreNull = false;
    while (!queue.isEmpty()) {
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
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
    int maxWidth = 0;
    Deque<TreeNode> queue = new LinkedList<>();
    root.val = 0;
    queue.offer(root);
    while (!queue.isEmpty()) {
      maxWidth = Math.max(maxWidth, queue.peekLast().val - queue.peekFirst().val + 1);
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        if (cur.left != null) {
          queue.offer(cur.left);
          cur.left.val = cur.val * 2 + 1;
        }
        if (cur.right != null) {
          queue.offer(cur.right);
          cur.right.val = cur.val * 2 + 2;
        }
      }
    }
    return maxWidth;
  }

  /**
   * 二叉树的最大深度，后序
   *
   * <p>扩展1，n 叉树，参考「N叉树的最大深度」，下方改为遍历所有子结点即可
   *
   * @param root the root
   * @return int int
   */
  public int maxDepth(TreeNode root) {
    if (root == null) return 0;
    int maxDepth = 0;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        if (cur.left != null) queue.offer(cur.left);
        if (cur.right != null) queue.offer(cur.right);
      }
      maxDepth += 1;
    }
    return maxDepth;
  }

  /**
   * 翻转二叉树 / 二叉树的镜像，前序 / 逐一交换遍历的结点的左右子树
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
      TreeNode tmp = cur.left;
      cur.left = cur.right;
      cur.right = tmp;
      if (cur.left != null) queue.offer(cur.left);
      if (cur.right != null) queue.offer(cur.right);
    }
    return root;
  }
}

/** 比对多棵树 */
class MultiTrees {
  /**
   * 二叉树的最近公共祖先，结点互异，后序遍历 / 转换为相交链表
   *
   * <p>特判 & 剪枝，判断 p & q 均在左子树内 & 返回非空结点
   *
   * <p>存储所有结点的父结点，然后通过结点的父结点从 p 结点开始不断往上跳，并记录已经访问过的结点，再从 q 结点开始不断往上跳，如果碰到已经访问过的结点，则为所求
   *
   * <p>扩展1，三个结点，设先对 n1 & n2 求 lca，再将 lca & n3 求即为结果
   *
   * <p>扩展2，n 叉树，参考
   * http://www.oier.cc/%E6%B4%9B%E8%B0%B7p3379%E3%80%90%E6%A8%A1%E6%9D%BF%E3%80%91%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88%EF%BC%88lca%EF%BC%89/
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
   * 合并二叉树，前序
   *
   * @param r1 the r 1
   * @param r2 the r 2
   * @return tree node
   */
  public TreeNode mergeTrees(TreeNode r1, TreeNode r2) {
    if (r1 == null || r2 == null) return r1 == null ? r2 : r1;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(r1);
    queue.offer(r2);
    while (queue.size() > 0) {
      TreeNode node1 = queue.poll(), node2 = queue.poll();
      node1.val += node2.val;
      // 如果二者左都不为空，就放到队列中
      // 如果 r1 左空，就把 r2 左挂为前者左
      if (node1.left != null && node2.left != null) {
        queue.offer(node1.left);
        queue.offer(node2.left);
      } else if (node1.left == null) {
        node1.left = node2.left;
      }
      if (node1.right != null && node2.right != null) {
        queue.offer(node1.right);
        queue.offer(node2.right);
      } else if (node1.right == null) {
        node1.right = node2.right;
      }
    }
    return r1;
  }

  /**
   * 另一棵树的子树 / 树的子结构
   *
   * <p>特判匹配树 & 主树为空两种情况，isSameTree 中的两处特判可以去除，因为匹配树 & 主树均非空
   *
   * @param root the root
   * @param subRoot the sub root
   * @return boolean boolean
   */
  public boolean isSubtree(TreeNode root, TreeNode subRoot) {
    //    if (subRoot == null) return true;
    //    if (root == null) return false;
    //    return isSubtree(root.left, subRoot)
    //        || isSubtree(root.right, subRoot)
    //        || isSameTree(root, subRoot);
    if (subRoot == null) return false;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      TreeNode node = queue.poll();
      if (node.val == subRoot.val && isSameTree(node, subRoot)) {
        return true;
      }
      if (node.left != null) queue.offer(node.left);
      if (node.right != null) queue.offer(node.right);
    }
    return false;
  }

  /**
   * 翻转等价二叉树
   *
   * @param root1 the root 1
   * @param root2 the root 2
   * @return boolean
   */
  public boolean flipEquiv(TreeNode root1, TreeNode root2) {
    // 1.相等
    if (root1.equals(root2)) return true;
    // 2.仅其一空或值不等
    if (root1 == null || root2 == null || root1.val != root2.val) return false;
    // 3.同侧同时比较，再异侧
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
    //    if (p == null && q == null) return true;
    //    else if (p == null || q == null) return false;
    //    return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(p);
    queue.offer(q);
    while (!queue.isEmpty()) {
      TreeNode n1 = queue.poll(), n2 = queue.poll();
      if (n1 == null && n2 == null) continue;
      if (n1 == null || n2 == null || n1.val != n2.val) {
        return false;
      }
      // 顺序
      queue.offer(n1.left);
      queue.offer(n2.left);
      queue.offer(n1.right);
      queue.offer(n2.right);
    }
    return true;
  }
}

/**
 * 回溯，前序与后序结合，遵从如下规范
 *
 * <p>入参顺序为 selection, path, res(if need), ...args，其中 path 采用 stack 因为符合回溯的语义
 *
 * <p>按照子组列的顺序，建议按照表格记忆
 *
 * <p>剪枝必须做在进入回溯之前，如排序，或回溯选择分支之内，如分支选择非法而 break
 */
class BacktrackingCombinatorics {
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
      path.offerLast(nums[i]);
      backtracking1(nums, path, res, i + 1);
      path.pollLast();
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
      path.offerLast(candidates[i]);
      backtracking2(candidates, path, res, i, target - candidates[i]);
      path.pollLast();
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
      path.offerLast(candidates[i]);
      backtracking3(candidates, path, res, i + 1, target - candidates[i]);
      path.pollLast();
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
      path.offerLast(nums[i]);
      backtracking4(nums, path, res, visited);
      path.pollLast();
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
      path.offerLast(nums[i]);
      backtracking5(nums, path, res, visited);
      path.pollLast();
      visited[i] = false;
    }
  }
}

class BacktrackingSearch extends DDFS {
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
    path.offerLast(root.val);
    if (targetSum - root.val == 0 && root.left == null && root.right == null) {
      // path 全局唯一，须做拷贝
      res.add(new ArrayList<>(path));
      // return 前须重置
      path.pollLast();
      return;
    }
    backtracking0(root.left, path, res, targetSum - root.val);
    backtracking0(root.right, path, res, targetSum - root.val);
    // 递归完成以后，必须重置变量
    path.pollLast();
  }

  /**
   * 路径总和III，返回路径总数，但从任意点出发，回溯 & 前缀和，题设结点值唯一
   *
   * <p>node.val:从该点出发满足的路径总数，则任两点不会有重复的路径
   *
   * @param root the root
   * @param targetSum the target sum
   * @return int int
   */
  public int pathSumIII(TreeNode root, int targetSum) {
    Map<Long, Integer> presum = new HashMap<>() {};
    presum.put(0L, 1); // base case
    return backtracking9(root, presum, 0, targetSum);
  }

  private int backtracking9(TreeNode root, Map<Long, Integer> presum, long cur, int targetSum) {
    if (root == null) return 0;
    cur += root.val;
    int path = presum.getOrDefault(cur - targetSum, 0);
    presum.put(cur, presum.getOrDefault(cur, 0) + 1);
    path +=
        backtracking9(root.left, presum, cur, targetSum)
            + backtracking9(root.right, presum, cur, targetSum);
    presum.put(cur, presum.getOrDefault(cur, 0) - 1);
    return path;
  }

  /**
   * 括号生成，括号相关的参考「最长有效括号」与「有效的括号」
   *
   * @param n the n
   * @return list list
   */
  public List<String> generateParenthesis(int n) {
    List<String> res = new ArrayList<>();
    // 需要特判
    if (n > 0) backtracking7(n, n, new StringBuilder(), res);
    return res;
  }

  // 可选集为左右括号的剩余量
  private void backtracking7(int left, int right, StringBuilder path, List<String> res) {
    if (left == 0 && right == 0) {
      res.add(path.toString());
      return;
    }
    if (left > right) return;
    if (left > 0) {
      path.append('(');
      backtracking7(left - 1, right, path, res);
      path.deleteCharAt(path.length() - 1);
    }
    if (right > 0) {
      path.append(')');
      backtracking7(left, right - 1, path, res);
      path.deleteCharAt(path.length() - 1);
    }
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
    if (start == word.length() - 1) return board[r][c] == word.charAt(start);
    if (board[r][c] != word.charAt(start)) return false;

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
   * 二叉树的所有路径，前序，其实是回溯，由于 Java String immutable 才不需移除
   *
   * <p>BFS 解法参考「求根结点到叶子结点数字之和」分别维护一个结点与路径的队列
   *
   * @param root the root
   * @return list
   */
  public List<String> binaryTreePaths(TreeNode root) {
    List<String> res = new ArrayList<>();
    backtracking12(root, "", res);
    return res;
  }

  private void backtracking12(TreeNode root, String path, List<String> res) {
    if (root == null) return;
    if (root.left == null && root.right == null) {
      res.add(path + root.val);
      return;
    }
    String cur = path + root.val + "->";
    backtracking12(root.left, cur, res);
    backtracking12(root.right, cur, res);
  }
}

class BacktrackingElse extends DDFS {
  private final String[] LetterMap = {
    " ", "*", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
  };

  /**
   * 将数字串拆分为多个不超过 k 的子串，输出所有可能路径
   *
   * @param s
   * @return
   */
  public List<List<Integer>> splitNumbers(String s, int K) {
    List<List<Integer>> res = new ArrayList<>();
    //    if (K > Integer.MAX_VALUE) return res;
    backtracking15(s, 0, new ArrayDeque<>(), res, K);
    return res;
  }

  private void backtracking15(
      String s, int start, Deque<Integer> path, List<List<Integer>> res, int K) {
    if (start == s.length()) {
      res.add(new ArrayList<>(path));
      return;
    }
    // 每轮只截到 K 的位数
    int num = 0;
    for (int i = start; i < s.length(); i++) {
      num = num * 10 + (s.charAt(i) - '0');
      if (num > K) break;
      path.offerLast(num);
      backtracking15(s, i + 1, path, res, K);
      path.pollLast();
    }
  }

  /**
   * 复原IP地址
   *
   * <p>参考
   * https://leetcode-cn.com/problems/restore-ip-addresses/solution/hui-su-suan-fa-hua-tu-fen-xi-jian-zhi-tiao-jian-by/
   *
   * @param s the s
   * @return list list
   */
  public List<String> restoreIpAddresses(String s) {
    List<String> res = new ArrayList<>();
    // 特判
    if (s.length() > 12 || s.length() < 4) return res;
    backtracking6(s, new ArrayDeque<>(4), res, 0, 4);
    return res;
  }

  private void backtracking6(
      String s, Deque<String> path, List<String> res, int start, int segment) {
    if (start == s.length()) {
      if (segment == 0) res.add(String.join(".", path));
      return;
    }
    // 每段只截取三位数
    for (int i = start; i < start + 3 && i < s.length(); i++) {
      // 当前段分配的位数不够，或分配的位数过多，或数字过大
      if (segment * 3 < s.length() - i || !isValidIpSegment(s, start, i)) {
        continue;
      }
      path.offerLast(s.substring(start, i + 1));
      backtracking6(s, path, res, i + 1, segment - 1);
      path.pollLast();
    }
  }

  private boolean isValidIpSegment(String s, int lo, int hi) {
    int len = hi - lo + 1;
    // 前导零剪枝
    if (len > 1 && s.charAt(lo) == '0') return false;
    int num = len > 0 ? Integer.parseInt(s.substring(lo, hi + 1)) : 0;
    return 0 <= num && num <= 255;
  }

  /**
   * 分割回文串
   *
   * <p>1.预处理所有子串的回文情况
   *
   * <p>2.暴力回溯
   *
   * <p>参考
   * https://leetcode-cn.com/problems/palindrome-partitioning/solution/hui-su-you-hua-jia-liao-dong-tai-gui-hua-by-liweiw/
   *
   * @param s the s
   * @return list
   */
  public List<List<String>> partition(String s) {
    List<List<String>> res = new ArrayList<>();
    int len = s.length();
    boolean[][] dp = new boolean[len][len];
    for (int i = 0; i < len; i++) {
      collect(s, i, i, dp);
      collect(s, i, i + 1, dp);
    }
    backtracking11(s, new ArrayDeque<>(), res, 0, dp);
    return res;
  }

  // 中心扩展，记录所有回文子串的始末点
  private void collect(String s, int lo, int hi, boolean[][] dp) {
    while (lo >= 0 && hi < s.length() && s.charAt(lo) == s.charAt(hi)) {
      dp[lo][hi] = true;
      lo -= 1;
      hi += 1;
    }
  }

  private void backtracking11(
      String s, Deque<String> path, List<List<String>> res, int start, boolean[][] dp) {
    if (start == s.length()) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = start; i < s.length(); i++) {
      if (!dp[start][i]) continue;
      path.offerLast(s.substring(start, i + 1));
      backtracking11(s, path, res, i + 1, dp);
      path.pollLast();
    }
  }

  /**
   * 电话号码的字母组合
   *
   * @param digits
   * @return
   */
  public List<String> letterCombinations(String digits) {
    if (digits == null || digits.length() == 0) return new ArrayList<>();
    List<String> res = new ArrayList<>();
    backtracking13(digits, new StringBuilder(), res, 0);
    return res;
  }

  private void backtracking13(String str, StringBuilder path, List<String> res, int idx) {
    if (idx == str.length()) {
      res.add(path.toString());
      return;
    }
    for (char ch : LetterMap[str.charAt(idx) - '0'].toCharArray()) {
      path.append(ch);
      backtracking13(str, path, res, idx + 1);
      path.deleteCharAt(path.length() - 1);
    }
  }

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

  /**
   * 验证IP地址
   *
   * <p>TODO
   *
   * @param queryIP
   * @return
   */
  // public String validIPAddress(String queryIP) {}
}
