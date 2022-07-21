// 字符串相加 留高位，最终翻转
// 大数相减 大减小就不用留高位，最终翻转，并移除前导零
// 字符串相乘 内外迭代，最终移除前导零

// 无重复字符的最长子串
// 最小覆盖子串
// 长度最小的子数组
// 至多包含K个不同字符的最长子串 上述模版一致，准备 window 和 counter 计数，注意处理缩窗的更新
// 最大连续1的个数III 贪心，将所有 0 都作为 1，触限则缩窗
// 绝对差不超过限制的最长连续子数组 维护两个单调队列，二者队头之差过限则缩窗出队
// 滑动窗口的最大值 一个单调队列，记录缩窗的索引与当前窗口内最大值，写入结果集

// 有效的括号 左入右出
// 有效的括号字符串 贪心维护左括号上下界，* 则 -+
// 字符串解码 左入右出，出则 repeat 更新，数字则添加，相当于「压缩字符串」的逆过程
// 简化路径 以 split 提取每段，回退则出栈，否则入栈，最终序列化栈内
// 验证栈序列 贪心维护两个序列的索引，push 序列去匹 pop 序列的
// 基本计算器

// 最长公共前缀 以第一个为参照物，垂直遍历
// 最长回文子串 中心扩散，内循环更新结果
// 回文子串 同上，内循环递增结果
// 至少有k个重复字符的最长子串 按字符统计个数后，用频率小于 k 的字符分割 s，再逐串递归取最大

// 字符串转换整数 判断正负，逐位写入时判断溢出
// 压缩字符串 读写双指针，读则加个内循环，然后逐位写入数字
// 翻转字符串里的单词 两次翻转，最终再移除首尾与单词间多余的空格

// 整数转罗马数字 从低位开始匹最大的字符，一直匹配直到取次大

// 每日温度
// 移掉k位数字 单调递减栈，记录值，并避免前导零，最终截取
// 下一个更大元素II 单调递减栈，记录索引
// 最大矩形
// 柱状图中最大的矩形

package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集字符串相关
 *
 * <p>拼接需求，只增不减则 String，否则选非线程安全的 StringBuilder 即可
 *
 * <p>对于 unicode 编码，字母与索引可以表示与 'a' 的偏移量，即 ch-'a'，而获取字母同理如 2+'a' is 'c'
 *
 * <p>取字符运算时需要 c-'0' 以隐式转为 int
 *
 * <p>遍历需求，s.charAt() 即可，需要更改则 s.toCharArray()
 *
 * @author cenghui
 */
public class SString extends DefaultSString {
  /**
   * 字符串相加，双指针同时遍历 & 比对 & 最后处理高位，模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * <p>类似包括合并 & 两数相加 & 大数相乘 & 大数相减 & 36 进制
   *
   * <p>扩展1，36 进制则先转 10 再转 36
   *
   * <p>扩展2，相减，同理维护一个高位，负责减，注意前导零，参下 reduceStrings
   *
   * <p>扩展3，其一为负，则提前判断首位再移除
   *
   * @param num1 the s 1
   * @param num2 the s 2
   * @return the string
   */
  public String addStrings(String num1, String num2) {
    final int BASE = 10; // 36 进制
    StringBuilder sum = new StringBuilder();
    int p1 = num1.length() - 1, p2 = num2.length() - 1;
    int carry = 0;
    while (p1 > -1 || p2 > -1 || carry != 0) { // 还要加上一个高位
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

  /**
   * 大数相减，尾插与借位，翻转与前导零
   *
   * <p>参考 https://mp.weixin.qq.com/s/RtAoA1hdf0h1PaVxRj_fzA
   *
   * @param num1
   * @param num2
   * @return
   */
  public String reduceStrings(String num1, String num2) {
    final int BASE = 10, l1 = num1.length(), l2 = num2.length();
    StringBuilder diff = new StringBuilder();
    // 1.预处理下方大减小，并判断符号
    if ((l1 == l2 && Integer.parseInt(num1) < Integer.parseInt(num2)) || l1 < l2) {
      String tmp = num1;
      num1 = num2;
      num2 = num1;
      diff.append('-');
    }
    // 2.从个位开始相减，注意借位，尾插，最终翻转
    int p1 = l1 - 1, p2 = l2 - 1;
    int carry = 0;
    while (p1 > -1 || p2 > -1) { // 由于保证大建小，因此不需要保留高位
      // 避免 n1-n2-carry < 0
      int n1 = p1 < 0 ? 0 : num1.charAt(p1) - '0',
          n2 = p2 < 0 ? 0 : num2.charAt(p2) - '0',
          tmp = (n1 - n2 - carry + BASE) % BASE;
      // res.insert(0, tmp) 则无需 reverse
      diff.append(tmp);
      carry = n1 - carry - n2 < 0 ? 1 : 0;
      p1 -= 1;
      p2 -= 1;
    }
    // 3.反转，移除前导零
    String str = diff.reverse().toString();
    int idx = frontNoBlank(str.toCharArray(), 0);
    return str.substring(idx, str.length());
  }

  // ASCII coding
  private char getChar(int num) {
    return num <= 9 ? (char) (num + '0') : (char) (num - 10 + 'a');
  }

  // ASCII 允许相减
  private int getInt(char num) {
    return '0' <= num && num <= '9' ? num - '0' : num - 'a' + 10;
  }

  /**
   * 字符串相乘，竖式，区分当前位和高位即可，最终需跳过前导零
   *
   * <p>扩展1，不能使用 vector 而是新建 string，使用加法，即每一轮内循环均产出一个字符串，外循环相加，不适合大数
   *
   * @param num1 the num 1
   * @param num2 the num 2
   * @return string string
   */
  public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0")) return "0";
    final int BASE = 10, l1 = num1.length(), l2 = num2.length();
    int[] product = new int[l1 + l2]; // 100*100=010000
    for (int p1 = l1 - 1; p1 > -1; p1--) {
      int n1 = num1.charAt(p1) - '0';
      for (int p2 = l2 - 1; p2 > -1; p2--) {
        int n2 = num2.charAt(p2) - '0', sum = product[p1 + p2 + 1] + n1 * n2;
        product[p1 + p2 + 1] = sum % BASE;
        product[p1 + p2] += sum / BASE;
      }
    }
    // 跳过前导零
    int idx = product[0] == 0 ? 1 : 0;
    StringBuilder res = new StringBuilder();
    while (idx < product.length) {
      res.append(product[idx]);
      idx += 1;
    }
    return res.toString();
  }

  /**
   * 比较版本号，逐个区间计数
   *
   * @param version1 the version 1
   * @param version2 the version 2
   * @return int int
   */
  public int compareVersion(String version1, String version2) {
    int l1 = version1.length(), l2 = version2.length(), p1 = 0, p2 = 0;
    while (p1 < l1 || p2 < l2) {
      int n1 = 0, n2 = 0;
      while (p1 < l1 && version1.charAt(p1) != '.') {
        n1 = n1 * 10 + version1.charAt(p1) - '0';
        p1 += 1;
      }
      // 跳过点号
      p1 += 1;
      while (p2 < l2 && version2.charAt(p2) != '.') {
        n2 = n2 * 10 + version2.charAt(p2) - '0';
        p2 += 1;
      }
      p2 += 1;
      if (n1 != n2) return n1 > n2 ? 1 : -1;
    }
    return 0;
  }

  /**
   * 第一个只出现一次的字符，对原串遍历两次
   *
   * <p>扩展1，第二个，下方找两次即可
   *
   * @param s the s
   * @return char char
   */
  public char firstUniqChar(String s) {
    int[] counter = new int[26];
    char[] chs = s.toCharArray();
    for (char ch : chs) {
      counter[ch - 'a'] += 1;
    }
    for (char ch : chs) {
      if (counter[ch - 'a'] == 1) return ch;
    }
    return ' ';
  }

  /**
   * 大数阶乘
   *
   * @param n
   * @return
   */
  //  public String factorial(String n) {
  //    int[] res = new int[10000];
  //    res[1] = 1; // 从array[1]开始
  //    int pos = 1; // point表示位数，刚开始只有一位array[1] 且 array[1] = 1，不能为0，0乘任何数为0
  //    int carry = 0; // carry表示进位数，刚开始进位为0
  //    int write = 0;
  //    // N的阶乘
  //    for (int i = 2; i <= numN; i++) {
  //      // 循环array[]，让每一位都与i乘
  //      for (write = 1; write <= pos; write++) {
  //        int tmp = res[write] * i + carry; // 表示不考虑进位的值
  //        carry = tmp / 10; // 计算进位大小
  //        res[write] = tmp % 10; // 计算本位值
  //      }
  //      // 处理最后一位的进位情况
  //      // 由于计算数组的最后一位也得考虑进位情况，所以用循环讨论
  //      // 因为可能最后一位可以进多位；比如 12 * 本位数8，可以进两位
  //      // 当进位数存在时，循环的作用就是将一个数分割，分割的每一位放入数组中
  //      while (carry > 0) {
  //        res[write] = carry % 10;
  //        write += 1; // 下一位
  //        carry /= 10;
  //      }
  //      pos = write - 1; // 由于上面while中循环有j++,所以位会多出一位，这里减去
  //    }
  //    return "";
  //  }
}

/**
 * 滑动窗口相关
 *
 * <p>https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/solution/hua-dong-chuang-kou-by-powcai/
 */
class WWindow {
  /**
   * 无重复字符的最长子串/最长无重复数组
   *
   * <p>扩展1，允许每种字符重复 k 次，下方「至少有k个重复字符的最长子串」允许有 k 种字符
   *
   * @param s the s
   * @return the int
   */
  public int lengthOfLongestSubstring(String s) {
    // ASCII 表总长
    int[] window = new int[128];
    int maxLen = 1, lo = 0, hi = 0;
    while (hi < s.length()) {
      char add = s.charAt(hi);
      window[add] += 1;
      while (window[add] > 1) {
        char out = s.charAt(lo);
        window[out] -= 1;
        lo += 1;
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
      hi += 1;
    }
    return s.length() < 1 ? 0 : maxLen;
  }

  /**
   * 最小覆盖字串，频率 & 计数 & 步进
   *
   * @param s main
   * @param t pattern
   * @return string string
   */
  public String minWindow(String s, String t) {
    int[] needle = new int[128];
    for (char ch : t.toCharArray()) {
      needle[ch] += 1;
    }
    // 遍历的指针与结果的始末
    int start = -1, end = s.length(), counter = t.length();
    int lo = 0, hi = 0;
    while (hi < s.length()) {
      char add = s.charAt(hi);
      if (needle[add] > 0) counter -= 1;
      needle[add] -= 1;
      hi += 1;
      while (counter == 0) {
        if (end - start > hi - lo) {
          start = lo;
          end = hi;
        }
        char out = s.charAt(lo);
        if (needle[out] == 0) counter += 1;
        needle[out] += 1;
        lo += 1;
      }
    }
    return end == s.length() ? "" : s.substring(start, end);
  }

  /**
   * 长度最小的子数组，和至少，最短，滑窗
   *
   * <p>扩展1，列出所有满足和为 target 的连续子序列，参考「和为k的子数组」，参下 annotate
   *
   * <p>扩展2，里边有负数，参考「和至少为k的最短子数组」或「表现良好的最长时间段」
   *
   * @param target the target
   * @param nums the nums
   * @return int int
   */
  public int minSubArrayLen(int target, int[] nums) {
    int winSum = 0, minLen = nums.length + 1;
    int lo = 0, hi = 0;
    while (hi < nums.length) {
      winSum += nums[hi]; // in
      while (winSum >= target) {
        minLen = Math.min(minLen, hi - lo + 1);
        winSum -= nums[lo]; // out
        lo += 1;
      }
      hi += 1;
    }
    //    for (int hi = 0; hi < nums.length; i++) {
    //      winSum += nums[hi];
    //      if (winSum == target) {
    //        // 入结果集
    //        continue;
    //      }
    //      while (winSum > target) {
    //        winSum -= nums[lo];
    //        lo += 1;
    //      }
    //    }
    return minLen == nums.length + 1 ? 0 : minLen;
  }

  /**
   * 至多包含K个不同字符的最长子串，类似「最小覆盖子串」
   *
   * @param s the s
   * @param k the k
   * @return int int
   */
  public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int maxLen = 0, counter = k;
    int lo = 0, hi = 0;
    int[] window = new int[128];
    char[] chs = s.toCharArray();
    while (hi < chs.length) {
      char add = chs[hi];
      window[add] += 1;
      if (window[add] == 1) counter -= 1;
      hi += 1;
      while (counter < 0) {
        char out = chs[lo];
        window[out] -= 1;
        if (window[out] == 0) counter += 1;
        lo += 1;
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
    }
    return maxLen;
  }

  /**
   * 最大连续1的个数III，返回最长子数组，最多可转换 k 个 0
   *
   * <p>统计窗口内 0 的个数，翻转所有扩窗的值为 1，次数上限后再缩窗
   *
   * <p>参考
   * https://leetcode.cn/problems/max-consecutive-ones-iii/solution/fen-xiang-hua-dong-chuang-kou-mo-ban-mia-f76z/
   *
   * @param nums
   * @param k
   * @return
   */
  public int longestOnes(int[] nums, int k) {
    int maxLen = 0, lo = 0, hi = 0;
    while (hi < nums.length) {
      if (nums[hi] == 0) k -= 1;
      while (k < 0) {
        if (nums[lo] == 0) k += 1;
        lo += 1;
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
      hi += 1;
    }
    return maxLen;
  }

  /**
   * 滑动窗口的最大值，单调队列，offer & max & poll
   *
   * @param nums the nums
   * @param k the k
   * @return int [ ]
   */
  public int[] maxSlidingWindow(int[] nums, int k) {
    int[] winMaxes = new int[nums.length - k + 1];
    Deque<Integer> mq = new ArrayDeque<>();
    for (int i = 0; i < nums.length; i++) {
      int n = nums[i];
      while (mq.size() > 0 && mq.peekLast() < n) mq.pollLast();
      mq.offerLast(n);
      if (i < k - 1) continue;
      // 缩窗的索引，当前窗口内最大值
      int outIdx = i - k + 1, winMax = mq.peekFirst();
      // segMax[outIdx] = mq.max();
      winMaxes[outIdx] = winMax;
      // mq.poll(nums[outIdx]);
      if (mq.size() > 0 && nums[outIdx] == winMax) mq.pollFirst();
    }
    return winMaxes;
  }

  /**
   * 绝对差不超过限制的最长连续子数组，数组内任意元素差有上限。
   *
   * <p>滑窗 & 双单调队列维护窗口内的最值 滑窗 R 右移的过程意味着新的值进窗口，维护更新单调队列
   *
   * <p>然后检查最值差值是否 <= k，如果 > 就 shrink L，同时也更新单调队列即可
   *
   * <p>参考
   * https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/solution/gai-zhuang-ban-hua-dong-chuang-kou-liang-271k/
   *
   * @param nums
   * @param limit
   * @return
   */
  public int longestSubarray(int[] nums, int limit) {
    Deque<Integer> maxMQ = new ArrayDeque<>(), minMQ = new ArrayDeque<>();
    int lo = 0, hi = 0, maxLen = 0;
    while (hi < nums.length) {
      int add = nums[hi];
      hi += 1;
      // 扩窗入队
      while (maxMQ.size() > 0 && add > maxMQ.peekLast()) maxMQ.pollLast();
      maxMQ.offerLast(add);
      while (minMQ.size() > 0 && add < minMQ.peekLast()) minMQ.pollLast();
      minMQ.offerLast(add);
      // 缩窗出队
      while (maxMQ.peekFirst() - minMQ.peekFirst() > limit) {
        int out = nums[lo];
        lo += 1;
        if (maxMQ.peekFirst() == out) maxMQ.pollFirst();
        if (minMQ.peekFirst() == out) minMQ.pollFirst();
      }
      maxLen = Math.max(maxLen, hi - lo);
    }
    return maxLen;
  }

  private static class MonotonicQueue {
    private final Deque<Integer> mq = new LinkedList<>();

    /**
     * Push.
     *
     * @param num the num
     */
    public void push(int num) {
      while (mq.size() > 0 && mq.getLast() < num) {
        mq.pollLast();
      }
      mq.offerLast(num);
    }

    /**
     * Pop.
     *
     * @param num the num
     */
    public void pop(int num) {
      if (mq.size() > 0 && mq.getFirst() == num) {
        mq.removeFirst();
      }
    }

    /**
     * Max int.
     *
     * @return the int
     */
    public int max() {
      return mq.getFirst();
    }
  }
}

/**
 * 收集栈相关，括号左入右出，入则清空，出则覆盖
 *
 * <p>关于 Java 模拟 stack 的选型
 * https://qastack.cn/programming/6163166/why-is-arraydeque-better-than-linkedlist
 */
class SStack {
  /**
   * 有效的括号，括号相关的参考「最长有效括号」与「括号生成」
   *
   * <p>扩展1，需保证优先级，如 {} 优先级最高即其 [{}] 非法，因此需要额外维护一个变量标识，入栈时判断，出入栈时更新，参下 annotate
   *
   * <p>扩展2，左括号可不以正确的任意闭合，如 ([)] 返回true，同时不能视作同一种即只统计数量，如 {{][}}
   * 非法，即放弃对顺序的要求，而只要求同种的数量，因此使用三个变量分别统计数目，无需栈
   *
   * <p>扩展3，( ) * 三种符号，参下「有效的括号字符串」
   *
   * @param s the s
   * @return the boolean
   */
  public boolean isValid(String s) {
    if (s.length() % 2 == 1) return false;
    Map<Character, Character> pairs = new HashMap<>(3);
    pairs.put(')', '(');
    pairs.put(']', '[');
    pairs.put('}', '{');
    Deque<Character> stack = new ArrayDeque<>();
    // int level = 0;
    for (char ch : s.toCharArray()) {
      // 左括号入栈
      // if ((priorities.indexOf(ch) + 1) % 3 > level) return false;
      // level = Math.max((priorities.indexOf(ch) + 1) % 3, level);
      if (!pairs.containsKey(ch)) {
        stack.offerLast(ch);
        continue;
      }
      // 右括号出栈并判断
      if (stack.isEmpty() || stack.peekLast() != pairs.get(ch)) return false;
      // level = Math.max((priorities.indexOf(stack.peek() + 1) % 3, level);
      stack.pollLast();
    }
    return stack.isEmpty();
    //    int curLeft1 = 0, curLeft2 = 0, curLeft3 = 0;
    //    for (char ch : s.toCharArray()) {
    //      int cnt = pairs.containsKey(ch) ? 1 : -1;
    //      if (ch == ')' || ch == '(') curLeft1 += cnt;
    //      else if (ch == '}' || ch == '{') curLeft2 += cnt;
    //      else if (ch == ']' || ch == '[') curLeft3 += cnt;
    //      if (curLeft1 < 0 || curLeft2 < 0 || curLeft3 < 0) return false;
    //    }
  }

  /**
   * 有效的括号字符串，贪心，或模拟
   *
   * <p>维护未匹配的左括号数量可能的上下界，尽可能保证其合法，遍历结束时，所有的左括号都应和右括号匹配即下界为 0
   *
   * <p>下界至少为0，且上界不能为负
   *
   * <p>参考
   * https://leetcode-cn.com/problems/valid-parenthesis-string/solution/gong-shui-san-xie-yi-ti-shuang-jie-dong-801rq/
   *
   * @param s the s
   * @return boolean boolean
   */
  public boolean checkValidString(String s) {
    int minCnt = 0, maxCnt = 0;
    for (char ch : s.toCharArray()) {
      if (ch == '(') {
        minCnt += 1;
        maxCnt += 1;
      } else if (ch == ')') {
        minCnt -= 1;
        maxCnt -= 1;
      } else if (ch == '*') {
        // 贪心，尽可能匹配右括号
        minCnt -= 1;
        maxCnt += 1;
      }
      if (minCnt < 0) minCnt = 0;
      if (minCnt > maxCnt) return false;
    }
    return minCnt == 0;
  }

  /**
   * 字符串解码，如 3[a]2[bc] to aaabcbc，类似压缩字符串 & 原子的数量
   *
   * <p>recursion 参考「基本计算器」
   *
   * @param s the s
   * @return string string
   */
  public String decodeString(String s) {
    int cnt = 0;
    StringBuilder str = new StringBuilder();
    Deque<Integer> cntStack = new ArrayDeque<>();
    Deque<String> strStack = new ArrayDeque<>();
    for (char ch : s.toCharArray()) {
      if (ch == '[') {
        cntStack.offerLast(cnt);
        strStack.offerLast(str.toString());
        cnt = 0;
        str.delete(0, str.length());
      } else if (ch == ']') {
        int preCnt = cntStack.pollLast();
        String preStr = strStack.pollLast();
        str = new StringBuilder(preStr + str.toString().repeat(preCnt));
      } else if (ch >= '0' && ch <= '9') cnt = cnt * 10 + (ch - '0');
      else str.append(ch);
    }
    return str.toString();
  }

  /**
   * 简化路径
   *
   * @param path the path
   * @return string
   */
  public String simplifyPath(String path) {
    Deque<String> stack = new ArrayDeque<>();
    for (String seg : path.split("/")) {
      if (seg.equals("") || seg.equals(".")) continue;
      else if (seg.equals("..")) stack.pollLast();
      else stack.offerLast(seg);
    }
    StringBuilder res = new StringBuilder(stack.size());
    for (String str : stack) {
      res.append('/');
      res.append(str);
    }
    return res.length() == 0 ? "/" : res.toString();
  }

  /**
   * 基本计算器II，思路类似「字符串解码」
   *
   * <p>单栈存放 oprations 参考
   * https://leetcode.cn/problems/basic-calculator/solution/ji-ben-ji-suan-qi-by-leetcode-solution-jvir/
   *
   * <p>不包含括号，则参考
   * https://leetcode.cn/problems/basic-calculator-ii/solution/ji-ben-ji-suan-qi-ii-by-leetcode-solutio-cm28/
   *
   * <p>扩展1，同时有括号与五种运算符，才使用双栈，否则建议单栈即可，前者参考 https://leetcode.cn/submissions/detail/318542073/
   *
   * @param s the s
   * @return int int
   */
  public int calculate(String s) {
    char[] chs = s.toCharArray();
    Deque<Integer> numStack = new ArrayDeque<>();
    int n = 0, len = chs.length;
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
    return sum;
  }

  /**
   * 判断两个字符串是否含义一致，二者只包含 (,),+,-,a-z 且保证字母不会连续，即合法的多元一次表达式
   *
   * <p>只有加减与括号，则展开括号并单栈暂存运算符即可，代码模板参下「基本计算器II」
   *
   * <p>参考
   * https://leetcode-cn.com/problems/basic-calculator/solution/ji-ben-ji-suan-qi-by-wo-yao-chu-qu-luan-nae94/
   *
   * @param s1 the s 1
   * @param s2 the s 2
   * @return boolean
   */
  public boolean isSame(String s1, String s2) {
    int[] counter1 = countLetter(s1.toCharArray()), counter2 = countLetter(s2.toCharArray());
    for (int i = 0; i < 26; i++) {
      if (counter1[i] != counter2[i]) return false;
    }
    return true;
  }

  private int[] countLetter(char[] chs) {
    int[] counter = new int[26];
    int sign = 1;
    Deque<Integer> opStack = new ArrayDeque<>();
    opStack.offerLast(sign);
    for (int i = 0; i < chs.length; i++) {
      char ch = chs[i];
      if (ch == '(') opStack.offerLast(sign);
      if (ch == ')') opStack.pollLast();
      if (ch == '+') sign = opStack.peekLast();
      if (ch == '-') sign = -opStack.peekLast();
      if (ch >= 'a' && ch <= 'z') counter[ch - 'a'] += sign;
      // 基本计算器
      //      if (ch >= '0' && ch <= '9') {
      //        long num = 0;
      //        while (i < chs.length && Character.isDigit(chs[i])) {
      //          num = num * 10 + chs[i] - '0';
      //          i += 1;
      //        }
      //        i -= 1;
      //        res += sign * num;
      //      }
    }
    return counter;
  }

  /**
   * 验证栈序列，无重复，贪心，原地模拟
   *
   * <p>将 pushed 队列中的每个数都 push 到栈中，同时检查这个数是不是 popped 序列中下一个要 pop 的值，如果是就把它 pop 出来。
   *
   * <p>最后，检查不是所有的该 pop 出来的值都是 pop
   *
   * <p>扩展1，入序列为 [1,n]，参下 annotate
   *
   * @param pushed the pushed
   * @param popped the popped
   * @return boolean
   */
  public boolean validateStackSequences(int[] pushed, int[] popped) {
    int stTop = 0, popIdx = 0; // 两个数组遍历的索引
    //    for (int add = 0; i < N; i++) {
    for (int add : pushed) {
      pushed[stTop] = add; // 入栈
      stTop += 1;
      // 出栈，直至空或序列不匹配
      while (stTop != 0 && pushed[stTop - 1] == popped[popIdx]) {
        stTop -= 1;
        popIdx += 1;
      }
    }
    return stTop == 0; // 是否还有未出栈的
  }

  /**
   * 移除无效的括号
   *
   * <p>当遇到左括号时，确认栈中左括号数量 <= 栈中右括号数量 + 尚未遍历的右括号数量
   *
   * <p>当遇到右括号时，确认栈中左括号数量 大于 栈中右括号数量
   *
   * @param s the s
   * @return string
   */
  public String minRemoveToMakeValid(String s) {
    int unusedRight = 0;
    char[] chs = s.toCharArray();
    for (char ch : chs) {
      if (ch == ')') unusedRight += 1;
    }
    int write = 0, curLeft = 0, curRight = 0;
    for (char ch : chs) {
      if (ch == '(') {
        if (curLeft >= curRight + unusedRight) continue;
        curLeft += 1;
      } else if (ch == ')') {
        unusedRight -= 1;
        if (curLeft <= curRight) continue;
        curRight += 1;
      }
      chs[write] = ch;
      write += 1;
    }
    return String.valueOf(Arrays.copyOfRange(chs, 0, write));
  }
}

/** 子串相关 */
class SSubString {
  /**
   * 最长公共前缀，纵向扫描
   *
   * @param strs the strs
   * @return string string
   */
  public String longestCommonPrefix(String[] strs) {
    if (strs.length == 0) return ""; // 需要特判
    String ref = strs[0]; // 参照串
    for (int i = 0; i < ref.length(); i++) {
      char pivot = ref.charAt(i);
      for (int j = 1; j < strs.length; j++) { // 有 j 个字符需要比对
        if (i < strs[j].length() && strs[j].charAt(i) == pivot) continue;
        return ref.substring(0, i);
      }
    }
    return ref;
  }

  /**
   * 最长回文子串，中心扩展
   *
   * @param s the s
   * @return the string
   */
  public String longestPalindrome(String s) {
    char[] chs = s.toCharArray();
    int start = 0, end = 0, len = chs.length;
    for (int i = 0; i < 2 * len - 1; i++) {
      int lo = i / 2, hi = lo + i % 2;
      while (lo > -1 && hi < len && chs[lo] == chs[hi]) {
        if (hi - lo > end - start) {
          start = lo;
          end = hi;
        }
        lo -= 1;
        hi += 1;
      }
    }
    return String.valueOf(chs, start, end + 1);
  }

  /**
   * 回文子串，中心扩展
   *
   * @param s
   * @return
   */
  public int countSubstrings(String s) {
    char[] chs = s.toCharArray();
    int cnt = 0, len = chs.length;
    for (int i = 0; i < 2 * len - 1; i++) {
      int lo = i / 2, hi = lo + i % 2;
      while (lo > -1 && hi < len && chs[lo] == chs[hi]) {
        cnt += 1;
        lo -= 1;
        hi += 1;
      }
    }
    return cnt;
  }

  /**
   * 验证回文串，忽略空格与大小写，两侧聚拢
   *
   * @param s the s
   * @return boolean boolean
   */
  public boolean isPalindrome(String s) {
    char[] chs = s.toCharArray();
    int lo = 0, hi = s.length() - 1;
    while (lo < hi) {
      while (lo < hi && !Character.isLetterOrDigit(chs[lo])) lo += 1;
      while (lo < hi && !Character.isLetterOrDigit(chs[hi])) hi -= 1;
      if (lo >= hi) break;
      if (Character.toLowerCase(chs[lo]) != Character.toLowerCase(chs[hi])) return false;
      lo += 1;
      hi -= 1;
    }
    return true;
  }

  /**
   * 验证回文字符串II
   *
   * @param s
   * @return
   */
  public boolean validPalindrome(String s) {
    char[] chs = s.toCharArray();
    int lo = 0, hi = chs.length - 1;
    while (lo < hi) {
      // 分两种情况，一是右边减一，二是左边加一
      if (chs[lo] != chs[hi]) return isPalindrome(chs, lo, hi - 1) || isPalindrome(chs, lo + 1, hi);
      lo += 1;
      hi -= 1;
    }
    return true;
  }

  private boolean isPalindrome(char[] s, int lo, int hi) {
    while (lo < hi) {
      if (s[lo] != s[hi]) return false;
      lo += 1;
      hi -= 1;
    }
    return true;
  }

  /**
   * 至少有k个重复字符的最长子串，同个字符的个数。
   *
   * <p>分治，按字符统计个数后，用频率小于 k 的字符分割 s，再逐串判断递归判断。
   *
   * <p>参考
   * https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters/solution/jie-ben-ti-bang-zhu-da-jia-li-jie-di-gui-obla/
   *
   * @param s the s
   * @param k the k
   * @return int
   */
  public int longestSubstring(String s, int k) {
    if (s.length() < k) return 0; // 特判
    int[] counter = new int[26];
    char[] chs = s.toCharArray();
    for (char ch : chs) counter[ch - 'a'] += 1;
    for (char ch : chs) {
      // 题设选任意一个频率不足的字符
      if (counter[ch - 'a'] >= k) continue;
      // 将原串以该字符分割，分别获取子串内的最优解
      int maxLen = 0;
      for (String seg : s.split(String.valueOf(ch)))
        maxLen = Math.max(maxLen, longestSubstring(seg, k));
      return maxLen;
    }
    return s.length(); // 原字符串没有小于 k 的字符串
  }

  /**
   * 实现strStr()，返回 haystack 中匹配 needle 的首个子串的首个字符的索引
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/implement-strstr/solution/shua-chuan-lc-shuang-bai-po-su-jie-fa-km-tb86/
   *
   * @param haystack the haystack
   * @param needle the needle
   * @return int
   */
  public int strStr(String haystack, String needle) {
    char[] s1 = haystack.toCharArray(), s2 = needle.toCharArray();
    int l1 = s1.length, l2 = s2.length;
    // 枚举原串的发起点 O(n^2)
    for (int i = 0; i <= l1 - l2; i++) {
      // 否则从 s1[i+1] 开始一轮匹配
      int p1 = i, p2 = 0;
      while (p2 < l2 && s1[p1] == s2[p2]) {
        p1 += 1;
        p2 += 1;
      }
      // 从 s1[i] 开始能匹完 s2
      if (p2 == l2) return i;
    }
    return -1;
  }

  /**
   * 判断子序列，顺序满足即可。
   *
   * <p>扩展1，依次检查海量 s 是否均为 t 的子序列，假如长字符串的长度为 n，建立一个 n*26 大小的矩阵，表示每个位置上 26 个字符下一次出现的位置
   *
   * <p>参下
   * https://leetcode-cn.com/problems/is-subsequence/solution/dui-hou-xu-tiao-zhan-de-yi-xie-si-kao-ru-he-kuai-s/
   *
   * @param s pattern
   * @param t main
   * @return boolean boolean
   */
  public boolean isSubsequence(String s, String t) {
    char[] l1 = s.toCharArray(), l2 = t.toCharArray();
    int p1 = 0, p2 = 0;
    while (p1 < l1.length && p2 < l2.length) {
      if (l1[p1] == l2[p2]) p1 += 1;
      p2 += 1;
    }
    return p1 == s.length();
  }
}

/** 子串相关，「单词搜索」参考 TTree，「单词拆分」参考 DP */
class WWord extends DefaultSString {
  /**
   * 字符串转换整数，如 " -26" to 26
   *
   * <p>去空格 & 判正负 & 逐位加 & 判溢出
   *
   * <p>扩展1，包含浮点，参下 annotate
   *
   * @param s the s
   * @return the int
   */
  public int myAtoi(String s) {
    boolean isNegative = false;
    char[] chs = s.toCharArray();
    // 去首空格，并判正负
    int len = s.length(), idx = frontNoBlank(chs, 0);
    if (idx == len) return 0;
    if (chs[idx] == '-') isNegative = true;
    if (chs[idx] == '-' || chs[idx] == '+') idx += 1;
    // 从高位开始取，留意溢出
    int n = 0;
    for (int i = idx; i < len; i++) {
      char ch = chs[i];
      //      if (ch == '.') {
      //        int decimal = myAtoi(s.substring(i, len));
      //        return (num + decimal * Math.pow(0.1, len - i + 1)) * (isNegative ? -1 : 1);
      //      }
      if (ch < '0' || ch > '9') break;
      int pre = n;
      n = n * 10 + (ch - '0');
      if (pre != n / 10) return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
    }
    return n * (isNegative ? -1 : 1);
  }

  /**
   * 压缩字符串，原地字符串编码，如 [a,a,a,b,b] to [a,3,b,2]，前者即 "aaabb" 后者同理 "a3b2"
   *
   * <p>类似「移动零」与滑窗，前后指针分别作读写 & 统计重复区间 & 写入
   *
   * @param chars the chars
   * @return int int
   */
  public int compress(char[] chars) {
    // write & read
    int write = 0, read = 0, len = chars.length;
    while (read < len) {
      int start = read;
      while (start < len && chars[start] == chars[read]) start += 1;
      chars[write++] = chars[read];
      // 逐位写入数字
      if (start - read > 1) {
        char[] cnt = Integer.toString(start - read).toCharArray();
        for (char d : cnt) chars[write++] = d;
      }
      read = start;
    }
    //    return String.valueOf(Arrays.copyOfRange(chars, 0, lo + 1));
    return write;
  }

  /**
   * 翻转字符串里的单词，如 www.abc.com -> com.abc.www
   *
   * <p>参考
   * https://leetcode.cn/problems/reverse-words-in-a-string/solution/fan-zhuan-zi-fu-chuan-li-de-dan-ci-by-leetcode-sol/
   *
   * @param s the s
   * @return string string
   */
  public String reverseWords(String s) {
    char[] chs = s.toCharArray();
    int len = chs.length;
    // 翻转整个
    reverseChs(chs, 0, len - 1);
    // 带空格逐个翻转单词
    reverseEachWord(chs, len);
    // 移除单词间多余的空格
    return removeBlanks(chs, len);
  }

  private void reverseEachWord(char[] chs, int len) {
    int start = 0;
    while (start < len) {
      // 找到首字母
      int end = frontNoBlank(chs, start);
      // 末位置
      while (end < len && chs[end] != ' ') end += 1;
      reverseChs(chs, start, end - 1);
      start = end;
    }
  }

  private String removeBlanks(char[] chs, int len) {
    // 移除首空格
    int write = 0, read = frontNoBlank(chs, 0);
    // 单词间留一个空，并移除尾空格
    while (read < len) {
      // 找到首空格
      while (read < len && chs[read] != ' ') chs[write++] = chs[read++];
      // 找到尾空格，单词间留一个空
      read = frontNoBlank(chs, read);
      if (read == len) break;
      chs[write++] = ' ';
    }
    return String.valueOf(chs, 0, write);
  }

  /**
   * 单词接龙，返回 beginWord 每次 diff 一个字母，最终变为 endWord 的最短路径，且所有路径均包含在 wordList 内
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/word-ladder/solution/yan-du-you-xian-bian-li-shuang-xiang-yan-du-you-2/
   *
   * @param beginWord the begin word
   * @param endWord the end word
   * @param wordList the word list
   * @return int
   */
  public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    // 第 1 步：先将 wordList 放到哈希表里，便于判断某个单词是否在 wordList 里
    Set<String> wordSet = new HashSet<>(wordList);
    if (wordSet.size() == 0 || !wordSet.contains(endWord)) {
      return 0;
    }
    // 第 2 步：已经访问过的 word 添加到 visited 哈希表里
    Set<String> visited = new HashSet<>();
    // 分别用左边和右边扩散的哈希表代替单向 BFS 里的队列，它们在双向 BFS 的过程中交替使用
    Set<String> beginVisited = new HashSet<>();
    beginVisited.add(beginWord);
    Set<String> endVisited = new HashSet<>();
    endVisited.add(endWord);
    // 第 3 步：执行双向 BFS，左右交替扩散的步数之和为所求
    int step = 1;
    while (!beginVisited.isEmpty() && !endVisited.isEmpty()) {
      // 优先选择小的哈希表进行扩散，考虑到的情况更少
      if (beginVisited.size() > endVisited.size()) {
        Set<String> temp = beginVisited;
        beginVisited = endVisited;
        endVisited = temp;
      }
      // 逻辑到这里，保证 beginVisited 是相对较小的集合，nextLevelVisited 在扩散完成以后，会成为新的 beginVisited
      Set<String> nextLevelVisited = new HashSet<>();
      for (String word : beginVisited) {
        if (changeWordEveryOneLetter(word, endVisited, visited, wordSet, nextLevelVisited)) {
          return step + 1;
        }
      }
      // 原来的 beginVisited 废弃，从 nextLevelVisited 开始新的双向 BFS
      beginVisited = nextLevelVisited;
      step += 1;
    }
    return 0;
  }

  // 尝试对 word 修改每一个字符，看看能否落在 endVisited 中，扩展得到的新的 word 添加到 nextLevelVisited 里
  private boolean changeWordEveryOneLetter(
      String word,
      Set<String> endVisited,
      Set<String> visited,
      Set<String> wordSet,
      Set<String> nextLevelVisited) {
    char[] chs = word.toCharArray();
    for (int i = 0; i < word.length(); i++) {
      char originCh = chs[i];
      for (char curCh = 'a'; curCh <= 'z'; curCh++) {
        if (originCh == curCh) continue;
        chs[i] = curCh;
        String nextWord = String.valueOf(chs);
        if (wordSet.contains(nextWord)) continue;
        if (endVisited.contains(nextWord)) return true;
        if (visited.contains(nextWord)) continue;
        visited.add(nextWord);
        nextLevelVisited.add(nextWord);
      }
      chs[i] = originCh; // 恢复，下次再用
    }
    return false;
  }
}

/**
 * 进制转换，编码相关
 *
 * <p>num to str: 尽可能匹大大
 *
 * <p>str to num: multiply by multiple which the char represtants starting from the low order
 */
class CConvert extends DefaultSString {
  private final char[] CHARS = "0123456789ABCDEF".toCharArray();
  // 「整数转换英文表示」
  private final String[]
      num2str_small =
          {
            "Zero",
            "One",
            "Two",
            "Three",
            "Four",
            "Five",
            "Six",
            "Seven",
            "Eight",
            "Nine",
            "Ten",
            "Eleven",
            "Twelve",
            "Thirteen",
            "Fourteen",
            "Fifteen",
            "Sixteen",
            "Seventeen",
            "Eighteen",
            "Nineteen"
          },
      num2str_medium =
          {"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"},
      num2str_large =
          {
            "Billion", "Million", "Thousand", "",
          };

  /**
   * 数字转换为十六进制数，即十进制互转，上方为十六进制转换为数字
   *
   * <p>TODO 参考 https://juejin.cn/post/6844904058357022728
   *
   * @param num the num
   * @return string string
   */
  public String toHex(int num) {
    if (num == 0) return "0";
    StringBuilder res = new StringBuilder();
    long cur = num < 0 ? (long) (Math.pow(2, 32) + num) : num;
    while (cur > 0) {
      res.append(CHARS[(int) (cur % 16)]); // 取余
      cur /= 16; // 除以
    }
    // 10->16
    //    while (n > 0) {
    //      res += CHARS[n % 16];
    //      n /= 16;
    //    }
    // 16->10
    //    for (int i = 0; i < len; i++) {
    //      if (s[i] - 'a' >= 0 && s[i] - 'a' <= 5)
    //        sum += (long) (s[i] - 'a' + 10) * (long) pow(16.0, len - i - 1);
    //      else sum += (s[i] - '0') * (int) pow(16.0, len - i - 1);
    //    }
    return res.reverse().toString();
  }

  /**
   * Excel表列名称，十进制转 26
   *
   * <p>一般进制转换无须进行额外操作，是因为我们是在「每一位数值范围在 [0,x)」的前提下进行「逢 x 进一」。
   *
   * <p>但本题需要我们将从 1 开始，因此在执行「进制转换」操作前，我们需要先对 cn 减一，从而实现整体偏移
   *
   * @param cn the cn
   * @return string
   */
  public String convertToTitle(int cn) {
    StringBuilder res = new StringBuilder();
    int cur = cn - 1;
    while (cur > 0) {
      res.append((char) (cur % 26 + 'A'));
      cur = cur / 26 - 1;
    }
    return res.reverse().toString();
  }

  /**
   * 整数转罗马数字，贪心，匹最大的字符
   *
   * <p>扩展1，阿拉伯数字转中文，参考 https://www.nowcoder.com/practice/6eec992558164276a51d86d71678b300
   *
   * @param num the num
   * @return string string
   */
  public String intToRoman(int num) {
    // 保证遍历有序
    final int[] NUMs = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    final String[] ROMANs = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    StringBuilder roman = new StringBuilder();
    int cur = num;
    for (int i = 0; i < NUMs.length; i++) {
      int n = NUMs[i]; // 贪心，数字能匹配的最大值及其对应的罗马字符
      String ch = ROMANs[i];
      while (cur >= n) { // 一直匹配当前最大罗马字符，直到取次大
        roman.append(ch);
        cur -= n;
      }
    }
    return roman.toString();
  }

  /**
   * Excel表列序号，26 转十进制，类似罗马数字转整数
   *
   * @param ct the ct
   * @return int int
   */
  public int titleToNumber(String ct) {
    int sum = 0;
    for (char ch : ct.toCharArray()) sum = sum * 26 + (ch - 'A' + 1);
    return sum;
  }

  /**
   * 罗马数字转整数，累加求和，从高位，短开始匹
   *
   * <p>扩展1，汉字转阿拉伯数字
   *
   * <p>扩展2，IP 与 integer 互转，参下
   *
   * @param s the s
   * @return int int
   */
  public int romanToInt(String s) {
    Map<Character, Integer> mapping =
        new HashMap<>() {
          {
            put('I', 1);
            put('V', 5);
            put('X', 10);
            put('L', 50);
            put('C', 100);
            put('D', 500);
            put('M', 1000);
          }
        };
    int sum = 0;
    char[] chs = s.toCharArray();
    for (int i = 0; i < chs.length; i++) {
      int add = mapping.get(chs[i]);
      // IV 与 VI 否则常规的做法是累加即可
      if (i < chs.length - 1 && add < mapping.get(chs[i + 1])) sum -= add;
      else sum += add;
    }
    return sum;
  }

  /**
   * IPv4 与无符号十进制互转
   *
   * <p>参考 https://mp.weixin.qq.com/s/UWCuEtNS2kuAuDY-eIbghg
   *
   * @param IP the ip
   * @return long
   */
  public String convertIPInteger(String str) {
    final int N = 4;
    if (str.contains(".")) { // ipv4 -> int
      String[] fields = str.split("\\.");
      long num = 0;
      for (int i = 0; i < N; i++) num = num << 8 + Integer.parseInt(fields[i]);
      return String.valueOf(num);
    } else { // int -> ipv4
      long ipv4 = Long.parseLong(str);
      String num = "";
      for (int i = 0; i < N; i++) {
        num = ipv4 % 256 + "." + num;
        ipv4 /= 256;
      }
      return num.substring(0, num.length() - 1);
    }
  }

  /**
   * 进制转换，十进制转任意进制
   *
   * <p>https://www.nowcoder.com/practice/2cc32b88fff94d7e8fd458b8c7b25ec1?tpId=196&tqId=37170&rp=1&ru=/exam/oj&qru=/exam/oj&sourceUrl=%2Fexam%2Foj%3Ftab%3D%25E7%25AE%2597%25E6%25B3%2595%25E7%25AF%2587%26topicId%3D196%26page%3D1&difficulty=undefined&judgeStatus=undefined&tags=&title=
   *
   * @param num the num
   * @param radix the radix
   * @return string string
   */
  public String baseConvert(int num, int radix) {
    if (num == 0) return "0";
    StringBuilder res = new StringBuilder();
    boolean ngt = false;
    if (num < 0) {
      ngt = true;
      num *= -1;
    }
    while (num != 0) {
      res.append(CHARS[num % radix]);
      num /= radix;
    }
    if (ngt) res.append("-");
    return res.reverse().toString();
  }

  /**
   * 整数转换英文表示，iteratively
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/integer-to-english-words/solution/gong-shui-san-xie-zi-fu-chuan-da-mo-ni-b-0my6/
   *
   * @param num the num
   * @return string
   */
  public String numberToWords(int num) {
    if (num == 0) return num2str_small[0];
    StringBuilder str = new StringBuilder();
    for (int i = (int) 1e9, j = 0; i >= 1; i /= 1000, j++) {
      if (num < i) continue;
      str.append(num2Str(num / i) + num2str_large[j] + " ");
      num %= i;
    }
    while (str.charAt(str.length() - 1) == ' ') str.deleteCharAt(str.length() - 1);
    return str.toString();
  }

  private String num2Str(int x) {
    StringBuilder res = new StringBuilder();
    if (x >= 100) {
      res.append(num2str_small[x / 100] + " Hundred ");
      x %= 100;
    }
    if (x >= 20) {
      res.append(num2str_medium[x / 10] + " ");
      x %= 10;
    }
    if (x != 0) res.append(num2str_small[x] + " ");
    return res.toString();
  }
}

/** 单调栈，递增利用波谷剔除栈中的波峰，留下波谷，反之，波峰 */
class MonotonicStack {
  /**
   * 每日温度，单调栈，递减，即找到右边首个更大的数，与下方「下一个更大元素II」框架基本一致
   *
   * @param tem the t
   * @return int [ ]
   */
  public int[] dailyTemperatures(int[] tem) {
    Deque<Integer> stack = new ArrayDeque<>();
    int[] res = new int[tem.length];
    for (int i = 0; i < tem.length; i++) {
      // 更新 res[pre] 直到满足其数字超过 temperatures[i]
      while (!stack.isEmpty() && tem[i] > tem[stack.peekLast()]) {
        int preIdx = stack.pollLast();
        res[preIdx] = i - preIdx;
      }
      stack.offerLast(i);
    }
    return res;
  }

  /**
   * 下一个更大元素II，单调栈，题设循环数组因此下方取索引均需取余
   *
   * @param nums the nums
   * @return int [ ]
   */
  public int[] nextGreaterElements(int[] nums) {
    Deque<Integer> ms = new ArrayDeque<>();
    int len = nums.length;
    int[] res = new int[len];
    Arrays.fill(res, -1);
    for (int i = 0; i < 2 * len; i++) {
      int n = nums[i % len];
      while (!ms.isEmpty() && n > nums[ms.peekLast()]) res[ms.pollLast()] = n;
      ms.offerLast(i % len);
    }
    return res;
  }

  /**
   * 移掉k位数字，结果数值最小，单调栈 int n = 高位递增」的数，应尽量删低位。;
   *
   * <p>123531 这样「高位递增」的数，应尽量n
   *
   * <p>432135 这样「高位递减」的数，应尽量删高位，即让高位变小。
   *
   * <p>因此，如果当前遍历的数比栈顶大，符合递增，让它入栈。
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/remove-k-digits/solution/yi-zhao-chi-bian-li-kou-si-dao-ti-ma-ma-zai-ye-b-5/
   *
   * @param num the num
   * @param k the k
   * @return string string
   */
  public String removeKdigits(String num, int k) {
    if (num.length() == k) return "0";
    StringBuilder ms = new StringBuilder();
    for (char ch : num.toCharArray()) {
      // 单调递减栈即出栈直到栈顶 <= 方可入栈
      while (k > 0 && ms.length() > 0 && ms.charAt(ms.length() - 1) > ch) {
        ms.deleteCharAt(ms.length() - 1);
        k -= 1;
      }
      if (ch == '0' && ms.length() == 0) continue; // 避免前导零
      ms.append(ch);
    }
    String str = ms.substring(0, Math.max(ms.length() - k, 0));
    return str.length() == 0 ? "0" : str; // 没有删除足够 k
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
    int maxArea = 0, len = heights.length;
    // 整体向右移位，且首尾添加哨兵，下方不用判断边界
    int[] hs = new int[len + 2];
    System.arraycopy(heights, 0, hs, 1, len);
    Deque<Integer> ms = new ArrayDeque<>(); // 保存索引
    for (int i = 0; i < hs.length; i++) maxArea = pushAndReturn(hs, i, maxArea, ms);
    return maxArea;
  }

  /**
   * 最大矩形，遍历每行的高度，通过栈
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/maximal-rectangle/solution/yu-zhao-zui-da-ju-xing-na-ti-yi-yang-by-powcai/
   *
   * @param matrix
   * @return
   */
  public int maximalRectangle(char[][] matrix) {
    int maxArea = 0, len = matrix[0].length;
    int[] hs = new int[len + 2];
    for (char[] rows : matrix) {
      Deque<Integer> ms = new ArrayDeque<>();
      for (int i = 0; i < hs.length; i++) {
        if (i > 0 && i < len + 1) { // 哨兵内
          if (rows[i - 1] == '1') hs[i] += 1;
          else hs[i] = 0;
        }
        maxArea = pushAndReturn(hs, i, maxArea, ms);
      }
    }
    return maxArea;
  }

  // 放入单调递减栈，比较的是在 hs 内的取值，并返回更新后的最大面积
  // 面积的高为 hs[pollLast] 宽为弹出后与 i 的距离
  private int pushAndReturn(int[] hs, int i, int maxArea, Deque<Integer> ms) {
    while (!ms.isEmpty() && hs[ms.peekLast()] > hs[i]) {
      int cur = ms.pollLast(), pre = ms.peekLast();
      maxArea = Math.max(maxArea, (i - pre - 1) * hs[cur]);
    }
    ms.offerLast(i);
    return maxArea;
  }

  /**
   * 去除重复字母/不同字符的最小子序列，且要求之后的整体字典序最小
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
    // 栈内尚存的字母
    boolean[] visited = new boolean[26];
    // 记录每个字符出现的最后一个位置
    int[] lastIdxs = new int[26];
    for (int i = 0; i < s.length(); i++) lastIdxs[s.charAt(i) - 'a'] = i;
    for (int i = 0; i < s.length(); i++) {
      char cur = s.charAt(i);
      if (visited[cur - 'a']) continue;
      while (!stack.isEmpty() && cur < stack.getLast() && lastIdxs[stack.getLast() - 'a'] > i) {
        visited[stack.pollLast() - 'a'] = false;
      }
      stack.offerLast(cur);
      visited[cur - 'a'] = true;
    }
    StringBuilder res = new StringBuilder();
    for (char ch : stack) res.append(ch);
    return res.toString();
  }
}

/** The type Default s string. */
abstract class DefaultSString extends DefaultArray {
  /**
   * Reverse.
   *
   * @param sb the sb
   * @param left the left
   * @param right the right
   */
  protected void reverse(StringBuilder sb, int left, int right) {
    int lo = left, hi = right;
    while (lo < hi) {
      char tmp = sb.charAt(lo);
      sb.setCharAt(lo, sb.charAt(hi));
      sb.setCharAt(hi, tmp);
      lo += 1;
      hi -= 1;
    }
  }

  /**
   * Reverse string.
   *
   * @param chars the chars
   * @param left the left
   * @param right the right
   */
  protected void reverseChs(char[] chars, int left, int right) {
    int lo = left, hi = right;
    while (lo < hi) {
      char temp = chars[lo];
      chars[lo] = chars[hi];
      chars[hi] = temp;
      lo += 1;
      hi -= 1;
    }
  }

  protected int frontNoBlank(char[] chs, int start) {
    while (start < chs.length && chs[start] == ' ') start += 1;
    return start;
  }

  /**
   * 前导零
   *
   * @param chs
   * @param start
   * @return
   */
  protected int frontNoZero(char[] chs, int start) {
    while (start < chs.length && chs[start] == '0') start += 1;
    return start;
  }
}
