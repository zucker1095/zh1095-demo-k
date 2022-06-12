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
    StringBuilder res = new StringBuilder();
    int p1 = num1.length() - 1, p2 = num2.length() - 1;
    int carry = 0;
    while (p1 >= 0 || p2 >= 0 || carry != 0) {
      int n1 = p1 < 0 ? 0 : getInt(num1.charAt(p1)), n2 = p2 < 0 ? 0 : getInt(num1.charAt(p2));
      int tmp = n1 + n2 + carry;
      res.append(getChar(tmp % BASE));
      carry = tmp / BASE;
      p1 -= 1;
      p2 -= 1;
    }
    return res.reverse().toString();
  }

  /**
   * 大数相减
   *
   * <p>参考 https://leetcode.cn/circle/discuss/zVtfxd/
   *
   * @param num1
   * @param num2
   * @return
   */
  public String reduceStrings(String num1, String num2) {
    StringBuilder res = new StringBuilder();
    if (integerLess(num1, num2)) {
      String tmp = num1;
      num1 = num2;
      num2 = num1;
      res.append('-');
    }
    int p1 = num1.length() - 1, p2 = num2.length() - 1;
    int carry = 0;
    while (p1 >= 0 || p2 >= 0) {
      int n1 = p1 < 0 ? 0 : num1.charAt(p1) - '0', n2 = p2 < 0 ? 0 : num2.charAt(p2) - '0';
      int tmp = (n1 - n2 - carry + 10) % 10;
      res.append(tmp); // res.insert(0,tmp) 则无需 reverse
      carry = n1 - carry - n2 < 0 ? 1 : 0;
      p1 -= 1;
      p2 -= 1;
    }
    String str = res.reverse().toString();
    int idx = 0; // 前导零
    for (char ch : str.toCharArray()) {
      if (ch != '0') break;
      idx += 1;
    }
    return str.substring(idx, str.length());
  }

  private boolean integerLess(String num1, String num2) {
    return num1.length() == num2.length()
        ? Integer.parseInt(num1) < Integer.parseInt(num2)
        : num1.length() < num2.length();
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
    int[] product = new int[l1 + l2];
    for (int p1 = l1 - 1; p1 >= 0; p1--) {
      int n1 = num1.charAt(p1) - '0';
      for (int p2 = l2 - 1; p2 >= 0; p2--) {
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
}

/**
 * 滑动窗口相关
 *
 * <p>https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/solution/hua-dong-chuang-kou-by-powcai/
 */
class WWindow {
  /**
   * 无重复字符的最长子串 / 最长无重复数组
   *
   * <p>扩展1，允许重复 k 次，即字符的个数，下方「至少有k个重复字符的最长子串」指种类
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
      while (window[add] == 2) {
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
    // 遍历的指针与结果的始末
    int lo = 0, hi = 0;
    int start = 0, end = Integer.MAX_VALUE, counter = t.length();
    int[] needle = new int[128];
    for (char ch : t.toCharArray()) {
      needle[ch] += 1;
    }
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
    return end == Integer.MAX_VALUE ? "" : s.substring(start, end);
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
   * 至多包含K个不同字符的最长子串，三步同上
   *
   * @param s the s
   * @param k the k
   * @return int int
   */
  public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int lo = 0, hi = 0;
    int maxLen = 0, counter = k;
    int[] window = new int[128];
    char[] chs = s.toCharArray();
    while (hi < chs.length) {
      char add = chs[hi];
      if (window[add] == 0) counter -= 1;
      window[add] += 1;
      hi += 1;
      while (counter < 0) { // just contain one more different char
        char out = chs[lo];
        if (window[out] == 1) counter += 1;
        window[out] -= 1;
        lo += 1;
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
    }
    return maxLen;
  }

  /**
   * 滑动窗口的最大值，单调队列，双端实现，offer & max & poll
   *
   * @param nums the nums
   * @param k the k
   * @return int [ ]
   */
  public int[] maxSlidingWindow(int[] nums, int k) {
    int[] segMax = new int[nums.length - k + 1];
    Deque<Integer> mq = new LinkedList<>();
    for (int i = 0; i < nums.length; i++) {
      int n = nums[i];
      while (mq.size() > 0 && mq.peekLast() < n) {
        mq.pollLast();
      }
      mq.offerLast(n);
      if (i < k - 1) continue;
      int outIdx = i - k + 1, curMax = mq.peekFirst(); // 左边界
      segMax[outIdx] = curMax; // segMax[outIdx] = mq.max();
      // mq.poll(nums[outIdx]);
      if (mq.size() > 0 && nums[outIdx] == curMax) mq.pollFirst();
    }
    return segMax;
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
    int maxLen = 0;
    Deque<Integer> maxMQ = new ArrayDeque<>(), minMQ = new ArrayDeque<>();
    int lo = 0, hi = 0;
    while (hi < nums.length) {
      int add = nums[hi];
      while (!maxMQ.isEmpty() && add >= nums[maxMQ.peekLast()]) {
        maxMQ.pollLast();
      }
      maxMQ.addLast(hi);
      while (!minMQ.isEmpty() && add <= nums[minMQ.peekLast()]) {
        minMQ.pollLast();
      }
      minMQ.addLast(hi);
      while (Math.abs(nums[maxMQ.peekFirst()] - nums[minMQ.peekFirst()]) > limit) {
        lo += 1;
        if (maxMQ.peekFirst() < lo) maxMQ.pollFirst();
        if (minMQ.peekFirst() < lo) minMQ.pollFirst();
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
      hi += 1;
    }
    return maxLen;
  }

  /**
   * 最大连续1的个数III，返回最长子数组，最多可转换 k 个 0
   *
   * <p>统计窗口内 0 的个数，窗口内容忍 k 个 0 即假设其全 1
   *
   * <p>参考
   * https://leetcode.cn/problems/max-consecutive-ones-iii/solution/fen-xiang-hua-dong-chuang-kou-mo-ban-mia-f76z/
   *
   * @param nums
   * @param k
   * @return
   */
  public int longestOnes(int[] nums, int k) {
    int maxLen = 0, zeroNum = 0;
    int lo = 0, hi = 0;
    while (hi < nums.length) {
      if (nums[hi] == 0) zeroNum += 1;
      while (zeroNum > k) {
        if (nums[lo] == 0) zeroNum -= 1;
        lo += 1;
      }
      maxLen = Math.max(maxLen, hi - lo + 1);
      hi += 1;
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
        mq.removeLast();
      }
      mq.addLast(num);
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
 * 收集栈相关，括号左入右出，入则清空，出则赋值
 *
 * <p>关于 Java 模拟 stack 的选型
 * https://qastack.cn/programming/6163166/why-is-arraydeque-better-than-linkedlist
 */
class SStack {
  /**
   * 有效的括号，括号相关的参考「最长有效括号」与「括号生成」
   *
   * <p>扩展1，需保证优先级，如 {} 优先级最高即其 [{}] 非法，因此需要额外维护一个变量标识，在出入栈时更新，参下 annotate
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
    // 外层括弧定义一个 Anonymous Inner Class
    // 内层括弧上是一个 instance initializer block，在内部匿名类构造时被执行
    Map<Character, Character> pairs = new HashMap<>(3);
    pairs.put(')', '(');
    pairs.put(']', '[');
    pairs.put('}', '{');
    Deque<Character> stack = new ArrayDeque<>();
    // int level = 0;
    for (char ch : s.toCharArray()) {
      // if ((priorities.indexOf(ch) + 1) % 3 > level) return false;
      if (!pairs.containsKey(ch)) {
        stack.addLast(ch);
        // level = Math.max((priorities.indexOf(ch) + 1) % 3, level);
        continue;
      }
      // level = Math.max((priorities.indexOf(stack.peek() + 1) % 3, level);
      if (stack.isEmpty() || stack.getLast() != pairs.get(ch)) return false;
      stack.removeLast();
    }
    //    int curLeft1 = 0, curLeft2 = 0, curLeft3 = 0;
    //    for (char ch : s.toCharArray()) {
    //      int weight = pairs.containsKey(ch) ? 1 : -1;
    //      if (ch == ')' || ch == '(') curLeft1 += weight;
    //      else if (ch == '}' || ch == '{') curLeft2 += weight;
    //      else if (ch == ']' || ch == '[') curLeft3 += weight;
    //      if (curLeft1 < 0 || curLeft2 < 0 || curLeft3 < 0) return false;
    //    }
    return stack.isEmpty();
  }

  /**
   * 有效的括号字符串，贪心
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
    int minCount = 0, maxCount = 0;
    for (char ch : s.toCharArray()) {
      if (ch == '(') {
        minCount += 1;
        maxCount += 1;
      } else if (ch == ')') {
        minCount -= 1;
        maxCount -= 1;
      } else if (ch == '*') {
        minCount -= 1;
        maxCount += 1;
      }
      minCount = Math.max(minCount, 0);
      if (maxCount < 0) return false;
    }
    return minCount == 0;
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
    int curCount = 0;
    StringBuilder curStr = new StringBuilder();
    Deque<Integer> countStack = new ArrayDeque<>();
    Deque<String> strStack = new ArrayDeque<>();
    for (char ch : s.toCharArray()) {
      if (ch == '[') {
        countStack.addLast(curCount);
        strStack.addLast(curStr.toString());
        curCount = 0;
        curStr = new StringBuilder();
      } else if (ch == ']') {
        int preCount = countStack.removeLast();
        String preStr = strStack.removeLast();
        curStr = new StringBuilder(preStr + curStr.toString().repeat(preCount));
      } else if (ch >= '0' && ch <= '9') {
        curCount = curCount * 10 + (ch - '0');
      } else {
        curStr.append(ch);
      }
    }
    return curStr.toString();
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
    StringBuilder res = new StringBuilder();
    for (String str : stack) {
      res.append('/');
      res.append(str);
    }
    return res.length() == 0 ? "/" : res.toString();
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
    int[] counter1 = countLetter(s1), counter2 = countLetter(s2);
    for (int i = 0; i < 26; i++) {
      if (counter1[i] != counter2[i]) return false;
    }
    return true;
  }

  private int[] countLetter(String s) {
    int[] counter = new int[26];
    int sign = 1;
    Deque<Integer> opStack = new ArrayDeque<>();
    opStack.offerLast(sign);
    for (int i = 0; i < s.length(); i++) {
      char ch = Character.toLowerCase(s.charAt(i));
      if ('a' <= ch && ch <= 'z') counter[ch - 'a'] += sign;
      else if (ch == '(') opStack.offerLast(sign);
      else if (ch == ')') opStack.pollLast();
      else if (ch == '+') sign = opStack.peekLast();
      else if (ch == '-') sign = -opStack.peekLast();
      // 基本计算器
      //      else {
      //        long num = 0;
      //        while (i < chs.length && Character.isDigit(chs[i])) {
      //          num = num * 10 + chs[i] - '0';
      //          i += 1;
      //        }
      //        i -= 1;
      //        ret += sign * num;
      //      }
    }
    return counter;
  }

  /**
   * 基本计算器，思路类似「字符串解码」
   *
   * <p>单栈存放 oprations 参考
   * https://leetcode.cn/problems/basic-calculator/solution/ji-ben-ji-suan-qi-by-leetcode-solution-jvir/
   *
   * <p>不包含括号，则参考
   * https://leetcode.cn/problems/basic-calculator-ii/solution/ji-ben-ji-suan-qi-ii-by-leetcode-solutio-cm28/
   *
   * <p>扩展1，同时有括号与五种运算符，才使用双栈，否则建议单栈即可
   *
   * @param s the s
   * @return int int
   */
  public int calculate(String s) {
    Map<Character, Integer> priOps =
        new HashMap<>() {
          {
            put('-', 1);
            put('+', 1);
            put('*', 2);
            put('/', 2);
            put('%', 2);
            put('^', 3);
          }
        };
    Deque<Integer> numStack = new ArrayDeque<>(); // 所有的数字
    numStack.addLast(0); // 防止第一个数为负数
    Deque<Character> opStack = new ArrayDeque<>(); // 存放所有非数字的操作
    char[] chs = s.toCharArray();
    for (int i = 0; i < chs.length; i++) {
      char cur = chs[i];
      if (cur == ' ') continue;
      else if (cur == '(') {
        opStack.addLast(cur);
      } else if (cur == ')') {
        // 计算到最近一个左括号为止
        while (!opStack.isEmpty()) {
          if (opStack.peekLast() == '(') {
            opStack.pollLast();
            break;
          }
          calc(numStack, opStack);
        }
      } else if (Character.isDigit(cur)) {
        // 数字
        int num = 0, idx = i;
        // 将从 i 位置开始后面的连续数字整体取出，加入 nums
        while (idx < chs.length && Character.isDigit(chs[idx])) {
          num = num * 10 + (chs[idx] - '0');
          idx += 1;
        }
        numStack.addLast(num);
        i = idx - 1;
      } else {
        // 操作符
        if (i > 0 && (chs[i - 1] == '(' || chs[i - 1] == '+' || chs[i - 1] == '-')) {
          numStack.addLast(0);
        }
        // 有一个新操作要入栈时，先把栈内可以算的都算了
        // 只有满足「栈内运算符」比「当前运算符」优先级高/同等，才进行运算
        while (!opStack.isEmpty() && opStack.peekLast() != '(') {
          if (priOps.get(opStack.peekLast()) < priOps.get(cur)) break;
          calc(numStack, opStack);
        }
        opStack.addLast(cur);
      }
    }
    // 将剩余的计算完
    while (!opStack.isEmpty()) {
      calc(numStack, opStack);
    }
    return numStack.peekLast();
  }

  private void calc(Deque<Integer> numStack, Deque<Character> opStack) {
    if (opStack.isEmpty() || numStack.isEmpty() || numStack.size() < 2) return;
    int res = 0, cur = numStack.pollLast(), pre = numStack.pollLast();
    char op = opStack.pollLast();
    if (op == '+') res = pre + cur;
    else if (op == '-') res = pre - cur;
    else if (op == '*') res = pre * cur;
    else if (op == '/') res = pre / cur;
    else if (op == '^') res = (int) Math.pow(pre, cur);
    else if (op == '%') res = pre % cur;
    numStack.addLast(res);
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
    int stackTop = 0, popIdx = 0;
    //    for (int add = 0; i < N; i++) {
    for (int add : pushed) {
      pushed[stackTop] = add;
      stackTop += 1;
      while (stackTop != 0 && pushed[stackTop - 1] == popped[popIdx]) {
        stackTop -= 1;
        popIdx += 1;
      }
    }
    return stackTop == 0;
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
    // 原地重写
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
    String ref = strs[0];
    for (int i = 0; i < ref.length(); i++) {
      char pivot = ref.charAt(i);
      for (int j = 1; j < strs.length; j++) { // 有 j 个字符需要比对
        if (i == strs[j].length() || strs[j].charAt(i) != pivot) {
          return ref.substring(0, i);
        }
      }
    }
    return ref;
  }

  /**
   * 最长回文子串，中心扩散
   *
   * @param s the s
   * @return the string
   */
  public String longestPalindrome(String s) {
    char[] chs = s.toCharArray();
    int lo = 0, hi = 0;
    for (int i = 0; i < chs.length; i++) {
      int odd = findLongestPalindrome(chs, i, i),
          even = findLongestPalindrome(chs, i, i + 1),
          maxLen = Math.max(odd, even);
      if (maxLen > hi - lo) {
        lo = i - (maxLen - 1) / 2;
        hi = i + maxLen / 2;
      }
    }
    return s.substring(lo, hi + 1);
  }

  private int findLongestPalindrome(char[] chs, int lo, int hi) {
    while (-1 < lo && hi < chs.length && chs[lo] == chs[hi]) {
      lo -= 1;
      hi += 1;
    }
    return hi - lo - 1;
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
      while (lo < hi && !Character.isLetterOrDigit(chs[lo])) {
        lo += 1;
      }
      while (lo < hi && !Character.isLetterOrDigit(chs[hi])) {
        hi -= 1;
      }
      if (lo >= hi) break;
      if (Character.toLowerCase(chs[lo]) != Character.toLowerCase(chs[hi])) {
        return false;
      }
      lo += 1;
      hi -= 1;
    }
    return true;
  }

  /**
   * 至少有k个重复字符的最长子串，每种字符的频率不少于。
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
    for (char ch : chs) {
      counter[ch - 'a'] += 1;
    }
    for (char ch : chs) {
      // split origin string by first char whose frequency is less than k to divide & conquer
      if (counter[ch - 'a'] >= k) continue;
      int maxLen = 0;
      for (String seg : s.split(String.valueOf(ch))) {
        maxLen = Math.max(maxLen, longestSubstring(seg, k));
      }
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
    char[] chs = haystack.toCharArray(), chsNeedle = needle.toCharArray();
    int lenChs = chs.length, lenNeedle = chsNeedle.length;
    for (int i = 0; i <= lenChs - lenNeedle; i++) { // 枚举原串的发起点
      int p1 = i, p2 = 0; // 开始一轮匹配
      while (p2 < lenNeedle && chs[p1] == chsNeedle[p2]) {
        p1 += 1;
        p2 += 1;
      }
      if (p2 == lenNeedle) return i;
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
    char[] chs = s.toCharArray(), chsNeedle = t.toCharArray();
    int p1 = 0, p2 = 0;
    while (p1 < chs.length && p2 < chsNeedle.length) {
      if (chs[p1] == chsNeedle[p2]) p1 += 1;
      p2 += 1;
    }
    return p1 == s.length();
  }
}

/**
 * 进制转换，编码相关
 *
 * <p>num to str: greedy
 *
 * <p>str to num: multiply by multiple which the char represtants starting from the low order
 */
class CConvert {
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
    int idx = 0, len = s.length();
    boolean isNegative = false;
    char[] chs = s.toCharArray();

    while (idx < len && chs[idx] == ' ') {
      idx += 1;
    }
    if (idx == len) return 0;

    if (chs[idx] == '-') isNegative = true;
    if (chs[idx] == '-' || chs[idx] == '+') idx += 1;

    int num = 0;
    for (int i = idx; i < len; i++) {
      char ch = chs[i];
      //      if (ch == '.') {
      //        int decimal = myAtoi(s.substring(i, len));
      //        return (num + decimal * Math.pow(0.1, len - i + 1)) * (isNegative ? -1 : 1);
      //      }
      // 非数字
      if (ch < '0' || ch > '9') break;
      // 判溢出
      int pre = num;
      num = num * 10 + (ch - '0');
      if (pre != num / 10) return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
    }

    return num * (isNegative ? -1 : 1);
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
    int lo = 0, hi = 0, len = chars.length;
    while (hi < len) {
      int cur = hi;
      while (cur < len && chars[cur] == chars[hi]) {
        cur += 1;
      }
      chars[lo] = chars[hi];
      lo += 1;
      if (cur - hi > 1) {
        // 逐位写入
        char[] num = Integer.toString(cur - hi).toCharArray();
        for (char digit : num) {
          chars[lo] = digit;
          lo += 1;
        }
      }
      hi = cur;
    }
    //    return String.valueOf(Arrays.copyOfRange(chars, 0, lo + 1));
    return lo;
  }

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
    // 尽量匹配大的
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
      cur /= 26;
      cur -= 1;
    }
    return res.reverse().toString();
  }

  /**
   * 整数转罗马数字，greedy 尽可能先选出大的数字进行转换
   *
   * <p>扩展1，阿拉伯数字转汉字，数字先一一对应建映射，逢位加十百千万标识
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
      while (cur >= NUMs[i]) {
        roman.append(ROMANs[i]);
        cur -= NUMs[i];
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
    int n = 0;
    for (char ch : ct.toCharArray()) {
      n = n * 26 + (ch - 'A' + 1);
    }
    return n;
  }

  /**
   * 罗马数字转整数，遍历字符，从短开始匹
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
    int n = 0;
    char[] chs = s.toCharArray();
    for (int i = 0; i < chs.length; i++) {
      int add = mapping.get(chs[i]);
      // 罗马数字的规则
      if (i < chs.length - 1 && add < mapping.get(chs[i + 1])) n -= add;
      else n += add;
    }
    return n;
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
      long integer = 0;
      for (int i = 0; i < N; i++) {
        integer = integer << 8 + Integer.parseInt(fields[i]);
      }
      return String.valueOf(integer);
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
    boolean negative = false;
    if (num < 0) {
      negative = true;
      num *= -1;
    }
    while (num != 0) {
      res.append(CHARS[num % radix]);
      num /= radix;
    }
    if (negative) res.append("-");
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
    while (str.charAt(str.length() - 1) == ' ') {
      str.deleteCharAt(str.length() - 1);
    }
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

/** 子串相关，「单词搜索」参考 TTree，「单词拆分」参考 DP */
class WWord extends DefaultSString {
  /**
   * 翻转字符串里的单词，如 www.abc.com -> com.abc.www
   *
   * <p>保证空格只有单词间的单个，先翻转整个串，再逐个翻转
   *
   * <p>TODO 参考
   * https://leetcode.cn/problems/reverse-words-in-a-string/solution/fan-zhuan-zi-fu-chuan-li-de-dan-ci-by-leetcode-sol/
   *
   * @param s the s
   * @return string string
   */
  public String reverseWords(String s) {
    // 1.去除首尾空格 & 翻转整个串
    StringBuilder str = new StringBuilder(s.trim()).reverse();
    char[] chs = new char[0];
    int write = 0, readStart = 0;
    for (int readEnd = 0; readEnd < s.length(); readEnd++) {
      if (s.charAt(readEnd) != ' ') continue;

      for (int i = readEnd - 1; i >= write; i++) {
        chs[i] = chs[readEnd];
      }
    }

    return str.toString();
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
        if (wordSet.contains(nextWord)) {
          if (endVisited.contains(nextWord)) return true;
          if (!visited.contains(nextWord)) {
            nextLevelVisited.add(nextWord);
            visited.add(nextWord);
          }
        }
      }
      // 恢复，下次再用
      chs[i] = originCh;
    }
    return false;
  }
}

/** 单调栈，递增利用波谷剔除栈中的波峰，留下波谷，反之，波峰 */
class MonotonicStack {
  /**
   * 每日温度，单调栈，递减，即找到右边首个更大的数，与下方「下一个更大元素II」框架基本一致
   *
   * @param temperatures the t
   * @return int [ ]
   */
  public int[] dailyTemperatures(int[] temperatures) {
    Deque<Integer> stack = new ArrayDeque<>();
    int[] res = new int[temperatures.length];
    for (int i = 0; i < temperatures.length; i++) {
      // 更新 res[pre] 直到满足其数字超过 temperatures[i]
      while (!stack.isEmpty() && temperatures[i] > temperatures[stack.getLast()]) {
        int preIdx = stack.removeLast();
        res[preIdx] = i - preIdx;
      }
      stack.addLast(i);
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
    Deque<Integer> stack = new ArrayDeque<>();
    int len = nums.length;
    int[] res = new int[len];
    Arrays.fill(res, -1);
    for (int i = 0; i < 2 * len; i++) {
      while (!stack.isEmpty() && nums[i % len] > nums[stack.getLast()]) {
        res[stack.removeLast()] = nums[i % len];
      }
      stack.addLast(i % len);
    }
    return res;
  }

  /**
   * 移掉k位数字，结果数值最小，单调栈
   *
   * <p>123531 这样「高位递增」的数，应尽量删低位。
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
      while (k > 0 && ms.length() > 0 && ms.charAt(ms.length() - 1) > ch) {
        ms.deleteCharAt(ms.length() - 1);
        k -= 1;
      }
      if (ch == '0' && ms.length() == 0) continue; // 避免前导零
      ms.append(ch);
    }
    // 没有删除足够 k
    String str = ms.substring(0, Math.max(ms.length() - k, 0));
    return str.length() == 0 ? "0" : str;
  }

  /**
   * 去除重复字母 / 不同字符的最小子序列，且要求之后的整体字典序最小
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
    for (int i = 0; i < s.length(); i++) {
      lastIdxs[s.charAt(i) - 'a'] = i;
    }
    for (int i = 0; i < s.length(); i++) {
      char cur = s.charAt(i);
      if (visited[cur - 'a']) continue;
      while (!stack.isEmpty() && cur < stack.getLast() && lastIdxs[stack.getLast() - 'a'] > i) {
        visited[stack.removeLast() - 'a'] = false;
      }
      stack.addLast(cur);
      visited[cur - 'a'] = true;
    }
    StringBuilder res = new StringBuilder();
    for (char ch : stack) {
      res.append(ch);
    }
    return res.toString();
  }
}

/** The type Default s string. */
abstract class DefaultSString {
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
}
