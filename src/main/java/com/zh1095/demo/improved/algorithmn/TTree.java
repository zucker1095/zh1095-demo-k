// 路径总和 I~III BFS 记录结果，回溯保存路径，引入前缀和与计数的映射
// 求根节点到叶节点数字之和 BFS 记录累加结果
// 从前序与中序遍历序列构造二叉树 先建立中序的映射
// 二叉树的序列化与反序列化 都是递归，反序列维护一个 idx

// 矩阵中的最长递增路径 记忆化搜索，保存从每个坐标出发的最长长度
// 岛屿数量 记录 visited 并 dfs 返回数量
// 岛屿的最大面积 记录 visited 并每次 dfs 返回时更新
// 不同岛屿的数量 序列化路径并引入 Set 去重
// 二叉树中所有距离为 k 的结点 先收集父节点，再从节点分别从三个方向 dfs 传参 from 避免回环
// 被围绕的区域 从边界找 O 点 dfs 标记

// 验证二叉搜索树 中序保留 pre
// 二叉搜索树中的第k小的元素 中序保留每次递减
// 恢复二叉搜索树 中序依次找两个逆序点，第二个要找到最后一个

// 二叉搜索树与双向链表 记录 pre 尾插，注意首个节点，其余的依次补充两个指针即可
// 将有序数组转换为二叉搜索树 二分分治两个区间后连接即可
// 二叉搜索树的后序遍历序列 单调栈，记录 pre 逆序遍历，若出现非逆序则 false
// 二叉搜索树中的插入操作 遍历，直到首个节点的值更小

// 二叉树中的最大路径和 总和更新结果，返回其一
// 二叉树的直径 模板一致
// 平衡二叉树 dfs 分别返回高度比对，返回 -1 则表示存在不平衡
// 二叉树展开为链表 后序遍历，依次将左子树挂在根的右，并将根的右挂在左子树的右

// 对称二叉树 BFS 出队时比较，并按序入队左右节点
// 翻转二叉树 BFS 出队时交换左右节点
// 二叉树最大宽度 维护每个节点与根的距离，每轮出队相加队列首尾的距离
// 二叉树的最大深度 BFS 统计清空队列的次数

// LCA 多个判断条件
// 合并二叉树 BFS 合并值后，依次判断 r1 的左右子树是否空，非空则入队，否则驳接至 r1
// 相同的树
// 另一棵树的子树
// 翻转等价二叉树

// 子集
// 组合总和
// 全排列
// 括号生成 可选集为左右括号的剩余量，当左更多则剪枝
// 单词搜索 不匹配当前字符则剪枝，否则深入，通过 recStack 避免环路
// 二叉树的所有路径
// 复原IP地址 留意特殊的剪枝与 IP 地址合法判断
// 电话号码的字母组合 类似子集，每次进入下一个可选的分支内
// 分割回文串 先收集 isPalindrome[lo][hi] 以便剪枝

package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集树相关，扩展大部分与打印路径相关
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
 * @author cenghui
 */
public class TTree {
  private int res3 = 0; // 「求根节点到叶节点数字之和」

  /**
   * 中序遍历，前中后依次为 入（结果集）左右 左入右 入右左
   *
   * <p>模板参考
   * https://leetcode-cn.com/problems/binary-tree-postorder-traversal/solution/zhuan-ti-jiang-jie-er-cha-shu-qian-zhong-hou-xu-2/
   *
   * <p>思维参考
   * https://leetcode-cn.com/problems/binary-tree-paths/solution/tu-jie-er-cha-shu-de-suo-you-lu-jing-by-xiao_ben_z/
   *
   * <p>扩展1，N 叉树，参下 annotate
   *
   * @param root the root
   * @return the list
   */
  public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> res = new ArrayList<>();
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      //      cur = stack.pollLast();
      //      res.add(cur.val);
      //      for (TreeNode node : cur.chlidren){
      //        stack.offerLast(node);
      //      }
      while (cur != null) {
        //        res.add(cur.val); // 前序与后序
        stack.offerLast(cur);
        cur = cur.left; // 后序 right
      }
      cur = stack.pollLast();
      res.add(cur.val); // 仅中序
      cur = cur.right; // 后序 left
    }
    //    Collections.reverse(res); // 仅后序
    return res;
  }

  /**
   * 路径总和，从根出发要求达到叶，BFS/前序
   *
   * <p>打印路径则参下「路径总和II」回溯
   *
   * @param root the root
   * @param sum the sum
   * @return boolean boolean
   */
  public boolean hasPathSum(TreeNode root, int sum) {
    //    if (root == null) return false;
    //    int v = root.val;
    //    if (root.left == null && root.right == null) return v == sum;
    //    return hasPathSum(root.left, sum - v) || hasPathSum(root.right, sum - v);
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
   * 求根节点到叶节点数字之和，BFS 维护两个队列逐层相加/前序
   *
   * @param root the root
   * @return int int
   */
  public int sumNumbers(TreeNode root) {
    //    dfs12(root, 0);
    //    return res3;
    if (root == null) return 0;
    int sum = 0; // 题设不会越界，因此不采用 long
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
   * 二叉树的下一个节点，给定一棵树中任一结点，返回其中序遍历顺序的下一个结点，from 指向父
   *
   * <p>若 x 有右子树，则下一个节点为 x 右子树最左侧节点。否则，分为两种情况
   *
   * <p>若 x 是父节点的左孩子，则 x 的父节点就是 x 的下一个节点。
   *
   * <p>若 x 是父节点的右孩子，则沿着父节点向上，直到找到一个节点的父节点的左孩子是该节点，则该节点的父节点就是 x 的下一个节点。
   *
   * <p>参考 https://mp.weixin.qq.com/s/yewlHvHSilMsrUMFIO8WAA
   *
   * @param node the node
   * @return next next
   */
  public _TreeNode getNext(_TreeNode node) {
    // 若有右子树，则返回其最左结点
    if (node.right != null) {
      _TreeNode cur = node.right;
      while (cur.left != null) {
        cur = cur.left;
      }
      return cur;
    }
    // 否则，回溯父节点记 P，直到 P 的左子树是 node，返回 P
    _TreeNode cur = node;
    while (cur.from != null) {
      if (cur.from.left.equals(cur)) return cur.from;
      cur = cur.from;
    }
    // 回溯至 root 仍没找到
    return null;
  }

  private class _TreeNode {
    private _TreeNode left, right, from;
  }
}

/** 根据序列建树相关 */
class Build {
  /**
   * 构造二叉树，题设元素唯一，否则，存在多棵树
   *
   * <p>扩展1，根据前序和中序，输出后序，不能构造树，参考 https://blog.csdn.net/u011068702/article/details/51914220
   *
   * <p>扩展2，给一个随机数组，生成相应的二叉搜索树，先排序，参下「将有序数组转换为二叉搜索树」
   *
   * @param preorder the preorder
   * @param inorder the inorder
   * @return the tree node
   */
  public TreeNode buildTree(int[] preorder, int[] inorder) {
    Map<Integer, Integer> v2i = new HashMap<>(); // 节点值唯一，因此可以存储二者间的映射
    for (int i = 0; i < preorder.length; i++) v2i.put(inorder[i], i);
    return buildTree1(preorder, 0, preorder.length - 1, v2i, 0);
  }

  // 从前序与中序遍历序列构造二叉树/重建二叉树
  private TreeNode buildTree1(
      int[] preorder, int preLo, int preHi, Map<Integer, Integer> v2i, int inLo) {
    if (preLo > preHi) return null;
    TreeNode root = new TreeNode(preorder[preLo]);
    int idx = v2i.get(preorder[preLo]), cntL = idx - inLo;
    root.left = buildTree1(preorder, preLo + 1, preLo + cntL, v2i, inLo);
    root.right = buildTree1(preorder, preLo + cntL + 1, preHi, v2i, idx + 1);
    return root;
  }

  // 从中序与后序遍历序列构造二叉树
  private TreeNode buildTree2(
      int[] postrorder, int postLo, int postHi, Map<Integer, Integer> v2i, int inLo) {
    if (postLo > postHi) return null;
    TreeNode root = new TreeNode(postrorder[postHi]);
    int idx = v2i.get(postrorder[postHi]), cntL = idx - inLo;
    root.left = buildTree2(postrorder, postLo, postLo + cntL - 1, v2i, inLo);
    root.right = buildTree2(postrorder, postLo + cntL, postHi - 1, v2i, idx + 1);
    return root;
  }

  //  /**
  //   * 根据前序和后序遍历构造二叉树
  //   *
  //   * @param preorder
  //   * @param postorder
  //   * @return
  //   */
  //  public TreeNode constructFromPrePost(int[] preorder, int[] postorder) {
  //    Map<Integer, Integer> v2i = new HashMap<>();
  //    for (int i = 0; i < postorder.length; i++) v2i.put(postorder[i], i);
  //    return buildTree3(preorder, 0, postorder.length - 1, 0, preorder.length - 1, v2i);
  //  }
  //
  //  private TreeNode buildTree3(
  //      int[] preorder, int postLo, int postHi, int preLo, int preHi, Map<Integer, Integer> v2i) {
  //    if (postLo > postHi || preLo > preHi) return null;
  //    TreeNode root = new TreeNode(preorder[preLo]);
  //    int idx = v2i.get(preorder[preLo + 1]), cntL = preLo + idx - postLo + 1;
  //    root.left = buildTree3(preorder, postLo, idx, preLo + 1, cntL, v2i);
  //    root.right = buildTree3(preorder, idx + 1, postHi - 1, cntL + 1, preHi, v2i);
  //    return root;
  //  }

  private int postIdx = 0; // 「输出后序」

  /**
   * 输出后序，根据前序和中序
   *
   * <p>扩展1，反转后的后序，则调换子树递归的顺序即可。
   *
   * @param pre the pre
   * @param inLo the in lo
   * @param inHi the in hi
   * @param v2i the hm
   * @param postorder the postorder
   */
  public void getPostorder(
      int[] preorder, int preLo, int preHi, Map<Integer, Integer> v2i, int inLo, int[] postorder) {
    if (preLo > preHi) return;
    int root = preorder[preLo];
    int idx = v2i.get(root), cntL = idx - inLo;
    getPostorder(preorder, preLo + 1, preLo + cntL, v2i, inLo, postorder);
    getPostorder(preorder, preLo + cntL + 1, preHi, v2i, idx + 1, postorder);
    postorder[postIdx] = root;
    postIdx += 1;
  }

  /**
   * 二叉树的序列化与反序列化，以下选用前序
   *
   * <p>扩展1，N 叉树，记录子树个数，参考 https://zhuanlan.zhihu.com/p/109521420
   */
  public class Codec {
    private int idx; // 反序列化

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
      return traverse(data.split(","));
    }

    private TreeNode traverse(String[] nodes) {
      if (idx > nodes.length - 1) return null;
      String v = nodes[idx];
      idx += 1;
      //      String cnt = vals[idx++];
      if (v.equals("null")) return null;
      TreeNode root = new TreeNode(Integer.parseInt(v));
      root.left = traverse(nodes);
      root.right = traverse(nodes);
      //      root.children = new TreeNode[cnt];
      //      for (int i = 0; i < cnt; i++) {
      //        root.children[i] = traversal(vals);
      //      }
      return root;
    }
  }
}

/**
 * 深度优先搜索，注意 visited 与 recStack 的区别，前者与整个图对应，因此不会重置，而后者与遍历一条路径有关，遍历完后，遍历下一条路径前重置，参考
 * https://www.geeksforgeeks.org/detect-cycle-in-a-graph/
 *
 * <p>遍历图与矩阵通常都需要 recStack，类比 Goosip 协议，而 DFS 本身决定每个点的步进方向
 *
 * <p>对于树，按照遍历的次序，dfs 即选型前序遍历或后序，而回溯相当于同时前序与后序
 *
 * <p>回溯 & dfs 框架基本一致，但前者适用 tree 这类不同分支互不连通的结构，而后者更适合 graph 这类各个分支都可能连通的
 *
 * <p>因此后者为避免 loop 不需要回溯，比如下方 grid[i][j]=2 后不需要再恢复
 */
class DDFS {
  /** The Directions. */
  protected final int[][] DIRECTIONS = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  // 「二叉树中所有距离为k的结点」结果集
  private final List<Integer> res5 = new ArrayList<>();
  // 「二叉树中所有距离为k的结点」目标结点的父
  private TreeNode parent;

  /**
   * 路径总和III，返回路径总数，但从任意点出发，题设值不重复，前缀和
   *
   * <p>node.val:从该点出发满足的路径总数，则任两点不会有重复的路径
   *
   * <p>扩展1，打印路径，参考「二叉树中的最大路径和」follow-up
   *
   * @param root the root
   * @param targetSum the target sum
   * @return int int
   */
  public int pathSumIII(TreeNode root, int targetSum) {
    Map<Long, Integer> preSum2Cnt = new HashMap<>();
    preSum2Cnt.put(0L, 1); // base case
    return dfs14(root, preSum2Cnt, 0, targetSum);
  }

  private int dfs14(TreeNode root, Map<Long, Integer> preSum2Cnt, long sum, int target) {
    if (root == null) return 0;
    sum += root.val;
    //    path.add(root.val);
    //    if (cur == target) res.add(new ArrayList(path));
    // 实际运行改用 getOrDefault
    int cnt = preSum2Cnt.get(sum - target);
    preSum2Cnt.put(sum, preSum2Cnt.get(sum) + 1);
    cnt += dfs14(root.left, preSum2Cnt, sum, target) + dfs14(root.right, preSum2Cnt, sum, target);
    preSum2Cnt.put(sum, preSum2Cnt.get(sum) - 1);
    return cnt;
  }

  /**
   * 岛屿数量，即求连通路总数，原地标记代替 recStack
   *
   * <p>参考
   * https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/
   *
   * @param grid the grid
   * @return int int
   */
  public int numIslands(char[][] grid) {
    int cnt = 0;
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] != '1') continue;
        dfs1(grid, i, j);
        cnt += 1;
      }
    }
    return cnt;
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
   * 岛屿的最大面积，即返回最长路径
   *
   * @param grid the grid
   * @return int int
   */
  public int maxAreaOfIsland(int[][] grid) {
    int maxArea = 0;
    for (int r = 0; r < grid.length; r++) {
      for (int c = 0; c < grid[r].length; c++) {
        if (grid[r][c] != 1) continue;
        maxArea = Math.max(maxArea, dfs2(grid, r, c));
      }
    }
    return maxArea;
  }

  private int dfs2(int[][] grid, int r, int c) {
    if (!inArea(grid, r, c) || grid[r][c] == 0) return 0;
    grid[r][c] = 0; // marking to avoid loop
    int pathLen = 1;
    for (int[] dir : DIRECTIONS) {
      pathLen += dfs2(grid, r + dir[0], c + dir[1]);
    }
    return pathLen;
  }

  /**
   * 矩阵中的最长递增路径，记忆化搜索，记录从每个节点开始的最长的长度
   *
   * @param matrix the matrix
   * @return int int
   */
  public int longestIncreasingPath(int[][] matrix) {
    int ROW = matrix.length, COL = matrix[0].length;
    int[][] lens = new int[ROW][COL];
    int maxLen = 0;
    for (int r = 0; r < ROW; r++) {
      for (int c = 0; c < COL; c++) {
        maxLen = Math.max(maxLen, dfs3(matrix, r, c, lens));
      }
    }
    return maxLen;
  }

  private int dfs3(int[][] matrix, int r, int c, int[][] lens) {
    if (lens[r][c] != 0) return lens[r][c];
    lens[r][c] += 1;
    for (int[] dir : DIRECTIONS) {
      int nr = r + dir[0], nc = c + dir[1];
      if (!inArea(matrix, nr, nc) || matrix[nr][nc] <= matrix[r][c]) continue;
      lens[r][c] = Math.max(lens[r][c], dfs3(matrix, nr, nc, lens) + 1);
    }
    return lens[r][c];
  }

  /**
   * 二叉树中所有距离为k的结点
   *
   * <p>参考
   * https://leetcode.cn/problems/all-nodes-distance-k-in-binary-tree/solution/er-cha-shu-zhong-suo-you-ju-chi-wei-k-de-qbla/
   *
   * @param root the root
   * @param target the target
   * @param k the k
   * @return list list
   */
  public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
    List<Integer> vers = new ArrayList<>();
    if (root == null) return vers;
    // 题设节点数有限，值互异，且 O(1) 查找采用 Map
    Map<Integer, TreeNode> parents = new HashMap<>(500);
    collectParents(root, parents);
    // 为避免重复，递归时传入来源，在递归前比较目标结点是否与来源结点相同
    dfs17(target, null, k, parents, vers);
    return vers;
  }

  private void collectParents(TreeNode node, Map<Integer, TreeNode> parents) {
    if (node.left != null) {
      parents.put(node.left.val, node);
      collectParents(node.left, parents);
    }
    if (node.right != null) {
      parents.put(node.right.val, node);
      collectParents(node.right, parents);
    }
  }

  private void dfs17(
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

  /**
   * 不同岛屿的数量
   *
   * @param grid the grid
   * @return the int
   */
  public int numDistinctIslands(int[][] grid) {
    Set<String> paths = new HashSet<>(); // 通过字符串去重
    for (int i = 0; i < grid.length; i++) {
      for (int j = 0; j < grid[0].length; j++) {
        if (grid[i][j] != 1) continue;
        StringBuilder path = new StringBuilder();
        dfs5(grid, path, i, j, i, j);
        paths.add(path.toString());
      }
    }
    return paths.size();
  }

  private void dfs5(int[][] grid, StringBuilder path, int x, int y, int preX, int preY) {
    if (!inArea(grid, x, y) || grid[x][y] == 0) return;
    grid[x][y] = 0; // mark it
    path.append(x - preX); // 分别记录相对的横纵坐标
    path.append(y - preY);
    for (int[] dir : DIRECTIONS) dfs5(grid, path, x + dir[0], y + dir[1], preX, preY);
  }

  /**
   * 被围绕的区域，填充所有被 X 围绕的 O，因此标记和边界联通的 O 路径即可。
   *
   * <p>参考
   * https://leetcode.cn/problems/surrounded-regions/solution/bfsdi-gui-dfsfei-di-gui-dfsbing-cha-ji-by-ac_pipe/
   *
   * @param board
   */
  public void solve(char[][] board) {
    int ROW = board.length, COL = board[0].length;
    for (int r = 0; r < ROW; r++) {
      for (int c = 0; c < COL; c++) {
        if (board[r][c] == 'O' && (r == 0 || c == 0 || r == ROW - 1 || c == COL - 1))
          dfs16(board, r, c);
      }
    }
    for (int r = 0; r < ROW; r++) {
      for (int c = 0; c < COL; c++) {
        if (board[r][c] == 'O') board[r][c] = 'X';
        if (board[r][c] == '#') board[r][c] = 'O';
      }
    }
  }

  private void dfs16(char[][] board, int r, int c) {
    if (!inArea(board, r, c) || board[r][c] != 'O') return;
    board[r][c] = '#';
    for (int[] dir : DIRECTIONS) dfs16(board, r + dir[0], c + dir[1]);
  }

  /**
   * 最长同值路径，找到任意起点的一条路径，所有结点值一致
   *
   * @param root
   * @return
   */
  public int longestUnivaluePath(TreeNode root) {
    dfs15(root);
    return cnt;
  }

  private int cnt = 0; // 「最长同值路径」

  private int dfs15(TreeNode node) {
    if (node == null) return 0;
    // 如果存在左子节点和根节点同值，更新左最长路径，否则左最长路径为 0
    int l = dfs15(node.left), r = dfs15(node.right);
    if (node.left != null && node.left.val == node.val) l = dfs15(node.left) + 1;
    else l = 0;
    if (node.right != null && node.right.val == node.val) r = dfs15(node.right) + 1;
    else r = 0;
    cnt = Math.max(cnt, l + r);
    return Math.max(l, r);
  }

  protected boolean inArea(char[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }

  protected boolean inArea(int[][] board, int i, int j) {
    return 0 <= i && i < board.length && 0 <= j && j < board[0].length;
  }
}

/** 二叉搜索树，中序为主，模板与「中序遍历」一致 */
class BBSTInorder {
  /**
   * 验证二叉搜索树，模板参上「中序遍历」
   *
   * @param root the root
   * @return boolean
   */
  public boolean isValidBST(TreeNode root) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    double pre = -Double.MAX_VALUE; // Integer.MIN_VALUE 即可，此处仅为通过官方测试
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
   * 二叉搜索树中的第k小的元素，左中右，对 k 做减法
   *
   * <p>扩展1，二叉搜索树中的第k大的元素，右中左，参下 annotate
   *
   * <p>扩展2，常数空间，Morris
   *
   * @param root the root
   * @param k the k
   * @return int int
   */
  public int kthSmallest(TreeNode root, int k) {
    int cnt = k;
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.offerLast(cur);
        cur = cur.left; // right
      }
      cur = stack.pollLast();
      cnt -= 1;
      if (cnt == 0) return cur.val;
      cur = cur.right; // left
    }
    return -1; // impossible
  }

  /**
   * 恢复二叉搜索树，中序找逆序对，框架保持「中序遍历」
   *
   * <p>中序依次找一对错误结点并交换，注意第二个点要最后一个。
   *
   * @param root the root
   */
  public void recoverTree(TreeNode root) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode pre = new TreeNode(Integer.MIN_VALUE), cur = root;
    TreeNode n1 = null, n2 = null;
    while (cur != null || !stack.isEmpty()) {
      while (cur != null) {
        stack.offerLast(cur);
        cur = cur.left;
      }
      cur = stack.pollLast();
      // stop recording util the last wrong pair e.g. swaping 15 & 5 not 15 & 8
      //    10
      //  15   5
      // 3 8 13 16
      if (pre.val > cur.val && n1 == null) n1 = pre;
      if (pre.val > cur.val && n1 != null) n2 = cur;
      pre = cur;
      cur = cur.right;
    }
    int tmp = n1.val;
    n1.val = n2.val;
    n2.val = tmp;
  }

  /**
   * 二叉搜索树的最小绝对差，中序，框架等同上方「中序遍历」
   *
   * @param root the root
   * @return minimum difference
   */
  public int getMinimumDifference(TreeNode root) {
    Deque<TreeNode> stack = new ArrayDeque<>();
    TreeNode cur = root;
    // 相邻节点差值最小，因此 pre 存储指针
    int minDiff = Integer.MAX_VALUE, pre = minDiff;
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
      fillLeft(root);
    }

    /**
     * Next int.
     *
     * @return the int
     */
    public int next() {
      TreeNode nxt = stack.pollLast();
      fillLeft(nxt.right);
      return nxt.val;
    }

    private void fillLeft(TreeNode node) {
      TreeNode cur = node;
      while (cur != null) {
        stack.offerLast(cur);
        cur = cur.left;
      }
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
  // 「二叉搜索树与双向链表」中序中当前遍历的上个节点 & 链表头节点，后者不变。
  private TreeNode head, pre;

  /**
   * 二叉搜索树与双向链表，双向链表，即补充叶节点的指针，中序
   *
   * <p>扩展1，逆序，则右中左
   *
   * @param root the root
   * @return tree node
   */
  public TreeNode treeToDoublyList(TreeNode root) {
    if (root == null) return null;
    dfs7(root);
    // 此时 pre 指向尾，连接首尾
    head.left = pre;
    pre.right = head;
    return head;
  }

  // 左中右，尾插，每次只处理前驱即左子树
  private void dfs7(TreeNode root) {
    if (root == null) return;
    dfs7(root.left);
    if (pre == null) head = root; // 中序首个节点，即最左子树，指向链表的头节点
    else pre.right = root; // 尾插，同时补充 pre 的后驱与 cur 的前驱
    root.left = pre;
    pre = root;
    dfs7(root.right);
  }

  /**
   * 将有序数组转换为二叉搜索树/最小高度树，前序，类似双路快排，以升序数组的中间元素作 root
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
   * 二叉搜索树的后序遍历序列
   *
   * <p>后序遍历反过来，根-右-左，维护一个 root，初始为最大值。
   *
   * <p>维护一个单调递增栈，当碰到小于栈顶的时候，出栈，并用栈顶更新root
   *
   * <p>当碰见当前节点大于 root 时，返回 false，否则循环完成，返回 true
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/solution/dan-diao-di-zeng-zhan-by-shi-huo-de-xia-tian/
   *
   * @param postorder
   * @return
   */
  public boolean verifyPostorder(int[] postorder) {
    Deque<Integer> ms = new ArrayDeque<>();
    // 表示上一个根节点的元素，最后一个元素可以看成无穷大节点的左孩子
    int pre = Integer.MAX_VALUE;
    // 逆向遍历即翻转的先序遍历
    for (int i = postorder.length - 1; i > -1; i--) {
      int cur = postorder[i];
      // 左子树小
      if (cur > pre) return false;
      // 数组元素小于单调栈的元素了，表示往左子树走了，记录下上个根节点
      // 找到这个左子树对应的根节点，之前右子树全部弹出，因为不可能在往根节点的右子树走了
      while (!ms.isEmpty() && cur < ms.peekLast()) pre = ms.pollLast();
      ms.offerLast(cur);
    }
    return true;
  }

  /**
   * 有序链表转换二叉搜索树，同理「将有序数组转换为二叉搜索树」找中点作根，再分治左右子树
   *
   * @param head the head
   * @return tree node
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

  /**
   * 删除二叉搜索树中的结点，递归找 target & 右子最左接 target 左 & 驳接
   *
   * @param root the root
   * @param key the key
   * @return tree node
   */
  public TreeNode deleteNode(TreeNode root, int key) {
    if (root == null) return null;
    if (key < root.val) root.left = deleteNode(root.left, key);
    if (key > root.val) root.right = deleteNode(root.right, key);
    if (key == root.val) {
      if (root.left == null) return root.right;
      if (root.right == null) return root.left;
      // 左右均非空，则将根的左子树挂在，根的下一个节点的左，即右子树的最左子树的左
      TreeNode cur = root.right;
      while (cur.left != null) {
        cur = cur.left;
      }
      cur.left = root.left;
      return root.right;
    }
    return root;
  }
}

/** 后序相关，常见为统计，自顶向下的递归相当于前序遍历，自底向上的递归相当于后序遍历 */
class Postorder {
  private int maxSum = Integer.MIN_VALUE; // 「二叉树中的最大路径和」
  private String maxPath; // 「二叉树中的最大路径和」follow up 打印路径
  private int diameter = 0; // 「二叉树的直径」

  /**
   * 二叉树中的最大路径和，从任意结点出发，后序遍历，模板与「二叉树的直径」近乎一致
   *
   * <p>三步曲，先取单侧 & 更新双侧结果 & 返回单侧更大者
   *
   * <p>扩展1，打印路径，参考 https://blog.csdn.net/Ackerman2/article/details/119060128
   *
   * @param root the root
   * @return int int
   */
  public int maxPathSum(TreeNode root) {
    singleSide1(root);
    //    singleSide3(root);
    return maxSum;
  }

  private int singleSide1(TreeNode root) {
    if (root == null) return 0;
    int l = Math.max(0, singleSide1(root.left)), r = Math.max(0, singleSide1(root.right));
    maxSum = Math.max(maxSum, l + r + root.val); // update both
    return Math.max(l, r) + root.val; // return solo
  }

  private CntAndPath singleSide3(TreeNode root) {
    if (root == null) return new CntAndPath();
    CntAndPath l = singleSide3(root.left), r = singleSide3(root.right);
    if (l.cnt <= 0) {
      l.cnt = 0;
      l.path = "";
    } else {
      l.path = l.cnt + "->";
    }
    if (r.cnt <= 0) {
      r.cnt = 0;
      l.path = "";
    } else {
      r.path = "->" + r.cnt;
    }
    int lc = l.cnt, rc = r.cnt;
    String lp = l.path, rp = r.path;
    if (root.val + lc + rc > maxSum) { // update both
      maxSum = lc + root.val + rc;
      maxPath = lp + root.val + rp;
    }
    CntAndPath solo = new CntAndPath(); // return solo
    if (lc > rc) {
      solo.cnt = lc + root.val;
      solo.path = lp + root.val;
    } else {
      solo.cnt = root.val + rc;
      solo.path = root.val + rp;
    }
    return solo;
  }

  /**
   * 二叉树的直径，后序，类似「二叉树的最大路径和」
   *
   * <p>扩展1，N 叉树，即无向图求最长路径，参考「N叉树的直径」
   *
   * @param root the root
   * @return int int
   */
  public int diameterOfBinaryTree(TreeNode root) {
    singleSide2(root);
    return diameter - 1;
  }

  private int singleSide2(TreeNode root) {
    if (root == null) return 0;
    int l = Math.max(0, singleSide2(root.left)), r = Math.max(0, singleSide2(root.right));
    diameter = Math.max(diameter, l + r + 1);
    return Math.max(l, r) + 1;
  }

  /**
   * 平衡二叉树，后序
   *
   * @param root the root
   * @return boolean boolean
   */
  public boolean isBalanced(TreeNode root) {
    return getHeight(root) != -1;
  }

  // 平衡则返高度，否则 -1
  private int getHeight(TreeNode root) {
    if (root == null) return 0;
    int l = getHeight(root.left);
    if (l == -1) return -1;
    int r = getHeight(root.right);
    if (r == -1) return -1;
    return Math.abs(l - r) < 2 ? Math.max(l, r) + 1 : -1;
  }

  /**
   * 二叉树展开为链表，单向链表，后序遍历，但以前序的次序连接
   *
   * @param root the root
   */
  public void flatten(TreeNode root) {
    if (root == null) return;
    flatten(root.left);
    flatten(root.right);
    // 依次将左子树挂在根的右，并将根的右挂在左子树的右，即后驱
    TreeNode oldRight = root.right;
    root.right = root.left;
    root.left = null;
    TreeNode tail = root;
    while (tail.right != null) tail = tail.right;
    tail.right = oldRight;
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
   * @return list list
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
      for (TreeNode l : leftPath) {
        for (TreeNode r : rightPath) {
          TreeNode root = new TreeNode(i);
          root.left = l;
          root.right = r;
          path.add(root);
        }
      }
    }
    return path;
  }

  private class CntAndPath {
    int cnt;
    String path;
  }
}

/** 广度优先搜索 */
class BBFS {
  /**
   * 二叉树的层序遍历，递归实现，前序，记录 level 即可
   *
   * <p>扩展1，N 叉树
   *
   * @param root the root
   * @return list list
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
   * 二叉树的层序遍历II
   *
   * @param root the root
   * @return list
   */
  public List<List<Integer>> levelOrderBottom(TreeNode root) {
    List<List<Integer>> res = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();
    if (root != null) queue.offer(root);
    while (!queue.isEmpty()) {
      List<Integer> curLevel = new ArrayList<>(queue.size());
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        curLevel.add(cur.val);
        TreeNode l = cur.left, r = cur.right;
        if (l != null) queue.offer(l);
        if (r != null) queue.offer(r);
      }
      res.add(curLevel);
    }
    Collections.reverse(res);
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
    if (root != null) queue.offer(root);
    boolean isOdd = true;
    while (!queue.isEmpty()) {
      Deque<Integer> curLevel = new ArrayDeque<>(queue.size());
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        if (isOdd) curLevel.offerLast(cur.val);
        else curLevel.offerFirst(cur.val);
        TreeNode l = cur.left, r = cur.right;
        if (l != null) queue.offer(l);
        if (r != null) queue.offer(r);
      }
      res.add(new ArrayList<>(curLevel));
      isOdd = !isOdd;
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
    Deque<TreeNode> queue = new LinkedList<>();
    if (root != null) queue.offerLast(root);
    while (!queue.isEmpty()) {
      res.add(queue.peekLast().val);
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.pollFirst();
        TreeNode l = cur.left, r = cur.right;
        if (l != null) queue.offerLast(l);
        if (r != null) queue.offerLast(r);
      }
    }
    return res;
  }

  /**
   * 二叉树的完全性校验，是否完全二叉树，类似「验证二叉搜索树」
   *
   * @param root the root
   * @return boolean boolean
   */
  public boolean isCompleteTree(TreeNode root) {
    boolean preNull = false;
    Queue<TreeNode> queue = new LinkedList<>();
    if (root != null) queue.offer(root);
    while (!queue.isEmpty()) {
      for (int i = queue.size(); i > 0; i--) {
        TreeNode cur = queue.poll();
        if (cur == null) {
          preNull = true;
          continue;
        }
        if (preNull) return false;
        queue.add(cur.left);
        queue.add(cur.right);
      }
    }
    return true;
  }

  /**
   * 对称二叉树
   *
   * @param root
   * @return
   */
  public boolean isSymmetric(TreeNode root) {
    // 不足两个节点
    if (root == null || (root.left == null && root.right == null)) return true;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root.left);
    queue.add(root.right);
    while (queue.size() > 0) {
      TreeNode l = queue.poll(), r = queue.poll();
      if (l == null && r == null) continue;
      if (l == null || r == null) return false;
      if (l.val != r.val) return false;
      // 按序入队
      queue.offer(l.left);
      queue.offer(r.right);
      queue.offer(l.right);
      queue.offer(r.left);
    }
    return true;
  }

  /**
   * 翻转二叉树/二叉树的镜像，前序/逐一交换遍历的结点的左右子树
   *
   * @param root the root
   * @return tree node
   */
  public TreeNode invertTree(TreeNode root) {
    if (root == null) return null;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      TreeNode n = queue.poll(), tmp = n.left;
      n.left = n.right;
      n.right = tmp;
      if (n.left != null) queue.offer(n.left);
      if (n.right != null) queue.offer(n.right);
    }
    return root;
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
        TreeNode n = queue.poll();
        if (n.left != null) queue.offer(n.left);
        if (n.right != null) queue.offer(n.right);
        //        for (TreeNode node : cur.children) {
        //          queue.offer(node);
        //        }
      }
      maxDepth += 1;
    }
    return maxDepth;
  }

  /**
   * 二叉树最大宽度，原地修改，或引入 hash 记录，如 key=level+value
   *
   * @param root the root
   * @return int int
   */
  public int widthOfBinaryTree(TreeNode root) {
    if (root == null) return 0;
    root.val = 0;
    int maxWidth = 0;
    Deque<TreeNode> queue = new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
      maxWidth = Math.max(maxWidth, queue.peekLast().val - queue.peekFirst().val + 1);
      for (int i = queue.size(); i > 0; i--) {
        TreeNode n = queue.poll();
        if (n.left != null) {
          queue.offer(n.left);
          n.left.val = n.val * 2 + 1;
        }
        if (n.right != null) {
          queue.offer(n.right);
          n.right.val = n.val * 2 + 2;
        }
      }
    }
    return maxWidth;
  }
}

/** 比对多棵树 */
class MultiTrees {
  /**
   * 二叉树的最近公共祖先，结点互异，后序遍历/转换为相交链表
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
    // 特判 & 剪枝，判断 p & q 均在左子树内 & 返回非空结点
    if (root == null || root == p || root == q) return root;
    TreeNode l = lowestCommonAncestor(root.left, p, q);
    if (l != null && l != q && l != p) return l;
    TreeNode r = lowestCommonAncestor(root.right, p, q);
    if (l != null && r != null) return root;
    return l == null ? r : l;
  }

  /**
   * 任意两个节点之间的最短路径，两个点分别沿着 LCA DFS 即可
   *
   * @param root
   * @param p
   * @param q
   * @return
   */
  public int distBetween(TreeNode root, TreeNode p, TreeNode q) {
    TreeNode lca = lowestCommonAncestor(root, p, q);
    return dfs18(lca, p) + dfs18(lca, q);
  }

  // 返回 target 与根的距离
  private int dfs18(TreeNode root, TreeNode target) {
    if (root == null) return -1;
    if (root == target) return 0;
    int l = dfs18(root.left, target);
    if (l != -1) return l + 1;
    int r = dfs18(root.right, target);
    return r == -1 ? -1 : r + 1;
  }

  /**
   * 合并二叉树，将 r2 合并至 r1
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
    while (!queue.isEmpty()) {
      TreeNode n1 = queue.poll(), n2 = queue.poll();
      // 合并当前节点的值
      n1.val += n2.val;
      // 依次判断 r1 的左右子树
      // 若二者左都不为空，则均需要遍历二者左的子节点，因此入队
      if (n1.left != null && n2.left != null) {
        queue.offer(n1.left);
        queue.offer(n2.left);
      }
      // 若 r1 左空，就把 r2 左挂为前者左，即仅遍历 r2 左的子节点，否则 r1 左无改动
      if (n1.left == null) n1.left = n2.left;
      // r1 右同理
      if (n1.right != null && n2.right != null) {
        queue.offer(n1.right);
        queue.offer(n2.right);
      }
      if (n1.right == null) n1.right = n2.right;
    }
    return r1;
  }

  /**
   * 相同的树，前序，迭代选用 bfs
   *
   * @param r1 the p
   * @param r2 the q
   * @return boolean boolean
   */
  public boolean isSameTree(TreeNode r1, TreeNode r2) {
    //    if (p == null && q == null) return true;
    //    else if (p == null || q == null) return false;
    //    return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(r1);
    queue.offer(r2);
    while (!queue.isEmpty()) {
      TreeNode n1 = queue.poll(), n2 = queue.poll();
      if (n1 == null && n2 == null) continue;
      if (n1 == null || n2 == null || n1.val != n2.val) return false;
      // 按序，保证出队同侧
      queue.offer(n1.left);
      queue.offer(n2.left);
      queue.offer(n1.right);
      queue.offer(n2.right);
    }
    return true;
  }

  /**
   * 另一棵树的子树/树的子结构 isSubStructure
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
      if (node.val == subRoot.val && isSameTree(node, subRoot)) return true;
      if (node.left != null) queue.offer(node.left);
      if (node.right != null) queue.offer(node.right);
    }
    return false;
  }

  /**
   * 翻转等价二叉树
   *
   * @param r1 the root 1
   * @param r2 the root 2
   * @return boolean boolean
   */
  public boolean flipEquiv(TreeNode r1, TreeNode r2) {
    // 均空
    if (r1 == null && r2 == null) return true;
    // 仅其一空或值不等
    if (r1 == null || r2 == null || r1.val != r2.val) return false;
    // 均非空且值相同，则依次比较同侧与异侧
    return (flipEquiv(r1.left, r2.left) && flipEquiv(r1.right, r2.right))
        || (flipEquiv(r1.left, r2.right) && flipEquiv(r1.right, r2.left));
  }
}

/**
 * 回溯，前序与后序结合，入参遵循
 *
 * <p>次序 selection, path, res(if need), ...args，其中 path 采用 stack 因为符合回溯的语义
 *
 * <p>按照子组列的顺序，建议按照表格记忆，通过索引标识是否元素重复，通过排序再取标识标识值重复
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
    if (nums.length > 0) bt1(nums, new ArrayDeque<>(), res, 0); // 需要特判
    return res;
  }

  private void bt1(int[] nums, Deque<Integer> path, List<List<Integer>> res, int start) {
    res.add(new ArrayList<>(path));
    // 另起一条路径
    for (int i = start; i < nums.length; i++) {
      path.offerLast(nums[i]);
      // 不可选重复，因此当前路径下一步选下一个元素
      bt1(nums, path, res, i + 1);
      path.pollLast();
    }
  }

  /**
   * 组合总和I，可选重复，对于负数仍通用
   *
   * <p>扩展1，不能重复，组合总和II，参下 annotate
   *
   * @param candidates the candidates
   * @param target the target
   * @return the list
   */
  public List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> res = new ArrayList<>();
    if (candidates.length > 0) {
      Arrays.sort(candidates);
      bt2(candidates, new ArrayDeque<>(), res, 0, target);
    }
    return res;
  }

  private void bt2(
      int[] candidates, Deque<Integer> path, List<List<Integer>> res, int start, int target) {
    if (target == 0) res.add(new ArrayList<>(path));
    for (int i = start; i < candidates.length; i++) {
      if (candidates[i] > target) break;
      // 不可选重复则跳过
      //      if (i > start && candidates[i - 1] == candidates[i]) continue;
      path.offerLast(candidates[i]);
      // 可选重复，则当前路径下一步选当前元素
      bt2(candidates, path, res, i, target - candidates[i]);
      // 不可选重复，则当前路径下一步选下一个元素
      //      bt2(candidates, path, res, i + 1, target - candidates[i]);
      path.pollLast();
    }
  }

  /**
   * 全排列I，无重复
   *
   * <p>扩展1，有重复，全排列II，参下 annotate
   *
   * <p>扩展2，「字符串的排列」一致
   *
   * @param nums the nums
   * @return the list
   */
  public List<List<Integer>> permute(int[] nums) {
    //    char[] chs = s.toCharArray();
    //    Arrays.sort(chs);
    List<List<Integer>> res = new ArrayList<>();
    if (nums.length > 0) {
      //    Arrays.sort(nums);
      bt4(nums, new ArrayDeque<>(), res, new boolean[nums.length]);
      //    return res.toArray(new String[0]);
    }
    return res;
  }

  private void bt4(int[] nums, Deque<Integer> path, List<List<Integer>> res, boolean[] recStack) {
    if (path.size() == nums.length) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = 0; i < nums.length; i++) {
      if (recStack[i]) continue;
      // 不在当前路径上但重复，或在
      //      if (recStack[i] || (i > 0 && nums[i] == nums[i - 1] && !recStack[i - 1])) continue;
      recStack[i] = true;
      path.offerLast(nums[i]);
      bt4(nums, path, res, recStack);
      path.pollLast();
      recStack[i] = false;
    }
  }
}

/** The type Backtracking search. */
class BacktrackingSearch extends DDFS {
  /**
   * 路径总和II，从根出发要求达到叶，打印路径
   *
   * <p>扩展1，从任何路径，参考「路径总和III」
   *
   * @param root the root
   * @param targetSum the target sum
   * @return list list
   */
  public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
    List<List<Integer>> paths = new ArrayList<>();
    bt0(root, new ArrayDeque<>(), paths, targetSum);
    return paths;
  }

  private void bt0(TreeNode root, Deque<Integer> path, List<List<Integer>> res, int target) {
    if (root == null) return;
    path.offerLast(root.val);
    int sum = target - root.val;
    if (sum == 0 && root.left == null && root.right == null) {
      res.add(new ArrayList<>(path));
      return;
    }
    bt0(root.left, path, res, sum);
    bt0(root.right, path, res, sum);
    path.pollLast();
  }

  /**
   * 括号生成，括号相关的参考「最长有效括号」与「有效的括号」
   *
   * @param n the n
   * @return list list
   */
  public List<String> generateParenthesis(int n) {
    List<String> pts = new ArrayList<>();
    if (n > 0) bt7(n, n, new StringBuilder(), pts); // 需要特判
    return pts;
  }

  // 可选集为左右括号的剩余量
  private void bt7(int l, int r, StringBuilder path, List<String> res) {
    if (l == 0 && r == 0) {
      res.add(path.toString());
      return;
    }
    if (l > r) return;
    if (l > 0) {
      path.append('(');
      bt7(l - 1, r, path, res);
      path.deleteCharAt(path.length() - 1);
    }
    if (r > 0) {
      path.append(')');
      bt7(l, r - 1, path, res);
      path.deleteCharAt(path.length() - 1);
    }
  }

  /**
   * 单词搜索/矩阵中的路径
   *
   * @param board the board
   * @param word the word
   * @return boolean boolean
   */
  public boolean exist(char[][] board, String word) {
    int ROW = board.length, COL = board[0].length;
    char[] chs = word.toCharArray();
    boolean[][] recStack = new boolean[ROW][COL];
    for (int r = 0; r < ROW; r++) {
      for (int c = 0; c < COL; c++) {
        if (bt8(board, r, c, chs, 0, recStack)) return true;
      }
    }
    return false;
  }

  private boolean bt8(char[][] board, int r, int c, char[] word, int start, boolean[][] recStack) {
    if (start == word.length - 1) return board[r][c] == word[start];
    if (board[r][c] != word[start]) return false;
    recStack[r][c] = true;
    for (int[] dir : DIRECTIONS) {
      int nX = r + dir[0], nY = c + dir[1];
      if (!recStack[nX][nY]
          && inArea(board, nX, nY)
          && bt8(board, nX, nY, word, start + 1, recStack)) return true;
    }
    recStack[r][c] = false;
    return false;
  }

  /**
   * 二叉树的所有路径，回溯，由于 Java String immutable 才不需移除
   *
   * <p>BFS 解法参考「求根结点到叶子结点数字之和」分别维护一个结点与路径的队列
   *
   * @param root the root
   * @return list list
   */
  public List<String> binaryTreePaths(TreeNode root) {
    List<String> paths = new ArrayList<>();
    bt12(root, new StringBuilder(), paths);
    return paths;
  }

  private void bt12(TreeNode root, StringBuilder path, List<String> paths) {
    if (root == null) return;
    String add = (path.length() == 0 ? "" : "->") + String.valueOf(root.val);
    path.append(add);
    if (root.left == null && root.right == null) paths.add(path.toString());
    bt12(root.left, path, paths);
    bt12(root.right, path, paths);
    int addCnt = add.length(), len = path.length();
    path.delete(len - addCnt, len);
  }

  /**
   * 所有可能的路径，DFS
   *
   * @param graph the graph
   * @return list
   */
  public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    List<List<Integer>> paths = new ArrayList<>();
    Deque<Integer> path = new ArrayDeque<>();
    path.offerLast(0);
    bt3(graph, 0, path, paths);
    return paths;
  }

  private void bt3(int[][] graph, int start, Deque<Integer> path, List<List<Integer>> paths) {
    if (start == graph.length - 1) {
      paths.add(new ArrayList<>(path));
      return;
    }
    for (int nxt : graph[start]) {
      path.offerLast(nxt);
      bt3(graph, nxt, path, paths);
      path.pollLast();
    }
  }
}

/** The type Backtracking else. */
class BacktrackingElse extends DDFS {
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
    List<String> ips = new ArrayList<>();
    if (s.length() > 12 || s.length() < 4) return ips;
    bt6(s, new ArrayDeque<>(4), ips, 0, 4);
    return ips;
  }

  private void bt6(String s, Deque<String> path, List<String> res, int start, int cnt) {
    if (start == s.length()) {
      if (cnt == 0) res.add(String.join(".", path));
      return;
    }
    for (int i = start; i < start + 3 && i < s.length(); i++) {
      String cur = s.substring(start, i + 1);
      if (!isValidIP(cur) || i == start + 3 || cnt * 3 < s.length() - i) continue;
      path.addLast(cur);
      bt6(s, path, res, i + 1, cnt - 1);
      path.removeLast();
    }
  }

  private boolean isValidIP(String s) {
    if (s.length() == 1) return true;
    if (s.length() > 3 || s.charAt(0) == '0') return false;
    return Integer.parseInt(s) <= 255;
  }

  /**
   * 电话号码的字母组合，类似「子集」
   *
   * @param digits the digits
   * @return list
   */
  public List<String> letterCombinations(String digits) {
    // 需要特判
    if (digits.length() == 0) return new ArrayList<>();
    List<String> res = new ArrayList<>();
    bt13(digits, new StringBuilder(), res, 0);
    return res;
  }

  private final String[] LetterMap = {
    " ", "*", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
  }; // 「电话号码的字母组合」

  private void bt13(String str, StringBuilder path, List<String> res, int start) {
    if (start == str.length()) {
      res.add(path.toString());
      return;
    }
    for (char ch : LetterMap[str.charAt(start) - '0'].toCharArray()) {
      path.append(ch);
      bt13(str, path, res, start + 1);
      path.deleteCharAt(path.length() - 1);
    }
  }

  /**
   * 删除无效的括号
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/remove-invalid-parentheses/solution/shan-chu-wu-xiao-de-gua-hao-by-leetcode-9w8au/
   *
   * @param s
   * @return
   */
  public List<String> removeInvalidParentheses(String s) {
    char[] chs = s.toCharArray();
    // 确定删除最少的合法括号对数
    int l = 0, r = 0;
    for (char ch : chs) {
      if (ch == '(') l += 1;
      if (ch == ')') {
        if (r < l) r += 1;
        else maxRemR += 1; // 最多移除右括号数量
      }
    }
    // 最大合法括号对数
    maxPair = Math.min(l, r);
    // 最多移除左括号数量
    maxRemL = l > maxPair ? l - maxPair : 0;
    bt14(chs, 0, 0, 0, 0, 0, new StringBuilder());
    return new ArrayList<>(res);
  }

  private final Set<String> res = new HashSet<>();

  // 最多移除左括号数量、最多移除右括号数量、最大合法括号对数
  private int maxRemL, maxRemR, maxPair;

  // 对每个位置字符，考虑加入和删除两种情况，记录当前位置左右括号对数，删除的括号对数
  // 以下几种情况可以剪枝
  // 1.非法 r>l
  // 2.放入的括号数量>最大对数 l > maxPair || r > maxPair
  // 3.删除的括号数量>最大删除数量 remL > maxRemL || remR > maxRemR
  private void bt14(char[] chs, int start, int l, int r, int remL, int remR, StringBuilder path) {
    if (r > l || l > maxPair || r > maxPair || remL > maxRemL || remR > maxRemR) return;
    if (start == chs.length) {
      res.add(path.toString());
      return;
    }
    char ch = chs[start];
    path.append(ch);
    int nxt = start + 1;
    if (ch == '(') {
      bt14(chs, nxt, l + 1, r, remL, remR, path);
      path.deleteCharAt(path.length() - 1);
      bt14(chs, nxt, l, r, remL + 1, remR, path);
    } else if (ch == ')') {
      bt14(chs, nxt, l, r + 1, remL, remR, path);
      path.deleteCharAt(path.length() - 1);
      bt14(chs, nxt, l, r, remL, remR + 1, path);
    } else {
      bt14(chs, nxt, l, r, remL, remR, path);
      path.deleteCharAt(path.length() - 1);
    }
  }

  /**
   * 划分为 k 个相等的子集
   *
   * <p>参考
   * https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/solution/javadai-fan-hui-zhi-de-hui-su-fa-by-caipengbo/
   *
   * @param nums
   * @param k
   * @return
   */
  public boolean canPartitionKSubsets(int[] nums, int k) {
    int sum = 0, max = 0;
    for (int n : nums) {
      sum += n;
      if (max < n) max = n;
    }
    target = sum / k;
    return sum % k == 0 && max <= target && bt15(nums, 0, k, 0, new boolean[nums.length]);
  }

  private int target; // 「划分为 k 个相等的子集」

  private boolean bt15(int[] nums, int start, int k, int sum, boolean[] recStack) {
    if (k == 0) return true;
    if (sum == target) return bt15(nums, 0, k - 1, 0, recStack);
    for (int i = start; i < nums.length; i++) {
      int cur = sum + nums[i];
      if (recStack[i] || cur > target) continue;
      recStack[i] = true;
      if (bt15(nums, i + 1, k, cur, recStack)) return true;
      recStack[i] = false;
    }
    return false;
  }

  /**
   * 分割回文串，将字符串分割为多个回文子串，返回所有结果
   *
   * <p>对整个串做回文判断 & 暴力回溯
   *
   * <p>参考
   * https://leetcode-cn.com/problems/palindrome-partitioning/solution/hui-su-you-hua-jia-liao-dong-tai-gui-hua-by-liweiw/
   *
   * @param s the s
   * @return list list
   */
  public List<List<String>> partition(String s) {
    List<List<String>> paths = new ArrayList<>();
    int len = s.length();
    // isPalindrome[i][j] 表示 s[i][j] 是否回文
    boolean[][] isPld = new boolean[len][len];
    for (int i = 0; i < len; i++) {
      collect(s, i, i, isPld);
      collect(s, i, i + 1, isPld);
    }
    bt11(s, new ArrayDeque<>(), paths, 0, isPld);
    return paths;
  }

  // 中心扩展，记录所有回文子串的始末点
  private void collect(String s, int lo, int hi, boolean[][] isPld) {
    while (lo > -1 && hi < s.length() && s.charAt(lo) == s.charAt(hi)) {
      isPld[lo][hi] = true;
      lo -= 1;
      hi += 1;
    }
  }

  private void bt11(
      String s, Deque<String> path, List<List<String>> res, int start, boolean[][] isPld) {
    if (start == s.length()) {
      res.add(new ArrayList<>(path));
      return;
    }
    for (int i = start; i < s.length(); i++) {
      if (!isPld[start][i]) continue; // [start:i] 区间非回文
      path.offerLast(s.substring(start, i + 1));
      bt11(s, path, res, i + 1, isPld);
      path.pollLast();
    }
  }

  /**
   * 将数字串拆分为多个不超过 k 的子串，打印路径
   *
   * @param s the s
   * @param K the k
   * @return list
   */
  public List<List<Integer>> splitNumbers(String s, int K) {
    List<List<Integer>> paths = new ArrayList<>();
    bt5(s, 0, new ArrayDeque<>(), paths, K);
    return paths;
  }

  private void bt5(String s, int start, Deque<Integer> path, List<List<Integer>> res, int K) {
    if (start == s.length()) {
      res.add(new ArrayList<>(path));
      return;
    }
    // 每轮只截到 K 的位数
    int n = 0;
    for (int i = start; i < s.length(); i++) {
      n = n * 10 + (s.charAt(i) - '0');
      if (n > K) break;
      path.offerLast(n);
      bt5(s, i + 1, path, res, K);
      path.pollLast();
    }
  }

  /**
   * 验证IP地址
   *
   * <p>TODO 参考 https://leetcode.cn/problems/validate-ip-address/solution/by-ac_oier-s217/
   *
   * @param ip
   * @return
   */
  public String validIPAddress(String ip) {
    String nt = "Neither";
    return ip.contains(":") ? validIpv6(ip, nt) : validIpv4(ip, nt);
  }

  private String validIpv4(String ip, String nt) {
    if (ip.startsWith(".") || ip.endsWith(".") || ip.contains("..")) return nt;
    String[] segs = ip.split("\\.");
    if (segs.length != 4) return nt;
    for (String s : segs) {
      if (s.length() > 3) return nt;
      int num = 0;
      for (char ch : s.toCharArray()) {
        if (!Character.isDigit(ch)) return nt;
        num = num * 10 + (ch - '0');
      }
      if (num > 255 || (s.charAt(0) == '0' && s.length() > 1)) return nt;
    }
    return "IPv4";
  }

  private String validIpv6(String ip, String nt) {
    if (ip.startsWith(":") || ip.endsWith(":") || ip.contains("::")) return nt;
    String[] segs = ip.split(":");
    if (segs.length != 8) return nt;
    for (String s : segs) {
      if (s.length() == 0 || s.length() > 4) return nt;
      for (char ch : s.toCharArray()) {
        ch = Character.toLowerCase(ch);
        if (!Character.isDigit(ch) && (ch < 'a' || ch > 'f')) return nt;
      }
    }
    return "IPv6";
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
    bt10(board);
  }

  // 1.跳过原始数字
  // 2.位置放 k 是否合适，是则，找到合适一组立刻返回
  // 3.九个数都试完，说明该棋盘无解
  private boolean bt10(char[][] board) {
    for (int y = 0; y < 9; y++) {
      for (int x = 0; x < 9; x++) {
        if (board[y][x] != '.') continue;
        for (char k = '1'; k <= '9'; k++) {
          if (!isValidSudoku(y, x, k, board)) continue;
          board[y][x] = k;
          if (bt10(board)) return true;
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
