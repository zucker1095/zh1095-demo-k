package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集字符串相关，个人遵从如下原则，如果部分查找不到，则至 DDP
 *
 * <p>拼接需求，只增不减则 String，否则选非线程安全的 StringBuilder 即可
 *
 * <p>遍历需求，s.charAt() 即可，需要更改则 s.toCharArray()
 *
 * <p>取字符运算时需要 c-'0' 以隐式转为 int
 *
 * @author cenghui
 */
public class SString extends DefaultSString {
  /**
   * 字符串相加，类似包括合并 & 两数相加 & 大数相乘 & 大数相减 & 36 进制
   *
   * <p>模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * <p>扩展1，36 进制则先转 10 再转 36
   *
   * <p>扩展2，相减，同理维护一个高位，负责减，注意前导零
   *
   * <p>扩展3，其一为负，则提前判断首位再移除
   *
   * @param num1 the s 1
   * @param num2 the s 2
   * @return the string
   */
  public String addStrings(String num1, String num2) {
    final int base = 10; // 36 进制
    StringBuilder res = new StringBuilder();
    int p1 = num1.length() - 1, p2 = num2.length() - 1;
    int carry = 0;
    while (p1 >= 0 || p2 >= 0 || carry != 0) {
      int n1 = p1 >= 0 ? getInt(num1.charAt(p1)) : 0;
      int n2 = p2 >= 0 ? getInt(num2.charAt(p2)) : 0;
      int tmp = n1 + n2 + carry;
      carry = tmp / base;
      res.append(getChar(tmp) % base);
      p1 -= 1;
      p2 -= 1;
    }
    if (carry == 1) {
      res.append(1);
    }
    return res.reverse().toString();
  }

  private char getChar(int num) {
    return num <= 9 ? (char) (num + '0') : (char) (num - 10 + 'a');
  }

  private int getInt(char num) {
    return '0' <= num && num <= '9' ? num - '0' : num - 'a' + 10;
  }

  /**
   * 数字转换为十六进制数，即十进制互转，上方为十六进制转换为数字
   *
   * <p>补码
   *
   * @param num the num
   * @return string string
   */
  public String toHex(int num) {
    if (num == 0) {
      return "0";
    }
    long cur = num;
    final int hex = 16;
    StringBuilder res = new StringBuilder();
    if (cur < 0) {
      cur = (long) (Math.pow(2, 32) + cur);
    }
    while (cur != 0) {
      long u = cur % hex;
      char c = (char) (u + '0');
      if (u >= 10) {
        c = (char) (u - 10 + 'a');
      }
      res.append(c);
      cur /= hex;
    }
    return res.reverse().toString();
  }

  /**
   * 字符串相乘，竖式，区分当前位和高位即可，最终需跳过前导零
   *
   * @param num1 the num 1
   * @param num2 the num 2
   * @return string string
   */
  public String multiply(String num1, String num2) {
    // 特判
    if (num1.equals("0") || num2.equals("0")) {
      return "0";
    }
    int[] res = new int[num1.length() + num2.length()];
    for (int i = num1.length() - 1; i >= 0; i--) {
      int n1 = num1.charAt(i) - '0';
      for (int j = num2.length() - 1; j >= 0; j--) {
        int n2 = num2.charAt(j) - '0';
        int sum = res[i + j + 1] + n1 * n2;
        res[i + j + 1] = sum % 10;
        res[i + j] += sum / 10;
      }
    }
    // 跳过前导零
    int idx = res[0] == 0 ? 1 : 0;
    StringBuilder ans = new StringBuilder();
    while (idx < res.length) {
      ans.append(res[idx]);
      idx += 1;
    }
    return ans.toString();
  }

  /**
   * 最长公共前缀，纵向扫描
   *
   * @param strs the strs
   * @return string string
   */
  public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) {
      return "";
    }
    int count = strs.length;
    for (int i = 0; i < strs[0].length(); i++) {
      char pivot = strs[0].charAt(i);
      for (int j = 1; j < count; j++) {
        if (i == strs[j].length() || strs[j].charAt(i) != pivot) {
          return strs[0].substring(0, i);
        }
      }
    }
    return strs[0];
  }

  /**
   * 最长回文子串，中心扩散
   *
   * @param s the s
   * @return the string
   */
  public String longestPalindrome(String s) {
    if (s == null || s.length() < 1) return "";
    int start = 0, end = 0;
    for (int i = 0; i < s.length(); i++) {
      int odd = findLongestPalindrome(s, i, i), even = findLongestPalindrome(s, i, i + 1);
      int len = Math.max(odd, even);
      if (len > end - start) {
        start = i - (len - 1) / 2;
        end = i + len / 2;
      }
    }
    return s.substring(start, end + 1);
  }

  // 分别从 lo&hi 扩散，直到二者所在字符不同
  private int findLongestPalindrome(String s, int lo, int hi) {
    while (lo > -1 && hi < s.length() && s.charAt(lo) == s.charAt(hi)) {
      lo -= 1;
      hi += 1;
    }
    return hi - lo - 1;
  }

  /**
   * 验证回文串，忽略空格与大小写
   *
   * @param s the s
   * @return boolean boolean
   */
  public boolean isPalindrome(String s) {
    int lo = 0, hi = s.length() - 1;
    while (lo < hi) {
      while (lo < hi && !Character.isLetterOrDigit(s.charAt(lo))) {
        lo += 1;
      }
      while (lo < hi && !Character.isLetterOrDigit(s.charAt(hi))) {
        hi -= 1;
      }
      if (lo < hi) {
        if (Character.toLowerCase(s.charAt(lo)) != Character.toLowerCase(s.charAt(hi))) {
          return false;
        }
        lo += 1;
        hi -= 1;
      }
    }
    return true;
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
    int[] count = new int[26];
    char[] chs = s.toCharArray();
    for (char ch : chs) {
      count[ch - 'a'] += 1;
    }
    for (char ch : chs) {
      if (count[ch - 'a'] == 1) return ch;
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
   * <p>扩展1，不使用 HashMap 则使用数组代替，索引通过 ASCII 取
   *
   * <p>扩展2，允许重复 k 次，即字符的个数，下方「至少有k个重复字符的最长子串」指种类
   *
   * @param s the s
   * @return the int
   */
  public int lengthOfLongestSubstring(String s) {
    int[] window = new int[128];
    int lo = 0, hi = 0;
    int res = 1;
    while (hi < s.length()) {
      char add = s.charAt(hi);
      window[add] += 1;
      while (window[add] == 2) {
        char out = s.charAt(lo);
        window[out] -= 1;
        lo += 1;
      }
      res = Math.max(res, hi - lo + 1);
      hi += 1;
    }
    return s.length() < 1 ? 0 : res;
  }

  /**
   * 最小覆盖字串，判断频率 & 计数 & 步进
   *
   * @param s main
   * @param t pattern
   * @return string string
   */
  public String minWindow(String s, String t) {
    int lo = 0, hi = 0;
    int counter = t.length(), start = 0, end = Integer.MAX_VALUE;
    int[] need = new int[128];
    for (int i = 0; i < t.length(); i++) {
      need[t.charAt(i)] += 1;
    }
    while (hi < s.length()) {
      char add = s.charAt(hi);
      if (need[add] > 0) counter -= 1;
      need[add] -= 1;
      hi += 1;
      while (counter == 0) {
        if (end - start > hi - lo) {
          start = lo;
          end = hi;
        }
        char out = s.charAt(lo);
        if (need[out] == 0) counter += 1;
        need[out] += 1;
        lo += 1;
      }
    }
    return end == Integer.MAX_VALUE ? "" : s.substring(start, end);
  }

  /**
   * 至多包含K个不同字符的最长子串，三步同上
   *
   * @param s the s
   * @param k the k
   * @return int
   */
  public int lengthOfLongestSubstringKDistinct(String s, int k) {
    int lo = 0, hi = 0;
    int res = 0, counter = k;
    int[] window = new int[128];
    while (hi < s.length()) {
      char add = s.charAt(hi);
      if (window[add] == 0) counter -= 1;
      window[add] += 1;
      hi += 1;
      while (counter < 0) {
        char out = s.charAt(lo);
        if (window[out] == 1) counter += 1;
        window[out] -= 1;
        lo += 1;
      }
      res = Math.max(res, hi - lo + 1);
    }
    return res;
  }

  /**
   * 长度最小的子数组，满足和不少于 target，滑窗
   *
   * <p>不能使用滑窗，因为下方缩窗的条件是整体满足 >=target，但可能已经满足的局部无法被收入
   *
   * <p>扩展1，列出所有满足和为 target 的连续子序列，参考「和为k的子数组」，参下 annotate
   *
   * <p>扩展2，里边有负数，参考「和至少为k的最短子数组」
   *
   * @param target the target
   * @param nums the nums
   * @return int int
   */
  public int minSubArrayLen(int target, int[] nums) {
    int res = Integer.MAX_VALUE, sum = 0;
    int lo = 0, hi = 0;
    while (hi < nums.length) {
      sum += nums[hi];
      while (sum >= target) {
        res = Math.min(res, hi - lo + 1);
        sum -= nums[lo];
        lo += 1;
      }
      // 替换上一段 while
      //      if (sum == target) {
      //        // 入结果集
      //        hi += 1;
      //        continue;
      //      }
      //      while (sum > target) {
      //        sum -= nums[lo];
      //        if (sum == target) {
      //          // 入结果集
      //        }
      //        lo += 1;
      //      }
      hi += 1;
    }
    return res == Integer.MAX_VALUE ? 0 : res;
  }

  /**
   * 滑动窗口的最大值
   *
   * @param nums the nums
   * @param k the k
   * @return int [ ]
   */
  public int[] maxSlidingWindow(int[] nums, int k) {
    int[] res = new int[nums.length - k + 1];
    Deque<Integer> mq = new LinkedList<>();
    for (int i = 0; i < nums.length; i++) {
      //      mq.push(nums[i]);
      while (mq.size() > 0 && mq.getLast() < nums[i]) {
        mq.removeLast();
      }
      mq.addLast(nums[i]);
      if (i < k - 1) {
        continue;
      }
      //      res[i - k + 1] = mq.max();
      res[i - k + 1] = mq.getFirst();
      //      mq.pop(nums[i - k + 1]);
      if (mq.size() > 0 && mq.getFirst() == nums[i - k + 1]) {
        mq.removeFirst();
      }
    }
    return res;
  }

  /**
   * 至少有k个重复字符的最长子串，要求次数非种类，即每个字符均需要 k 次
   *
   * <p>分治，用频率小于 k 的字符作为切割点, 将 s 切割为更小的子串进行处理
   *
   * @param s
   * @param k
   * @return
   */
  public int longestSubstring(String s, int k) {
    // 特判
    if (s.length() < k) return 0;
    int[] counter = new int[128];
    for (int i = 0; i < s.length(); i++) {
      counter[s.charAt(i)] += 1;
    }
    for (int ch : counter) {
      if (counter[ch] >= k) {
        continue;
      }
      // 找到次数少于 k 的字符串，则切分为多个小段分治
      int res = 0;
      for (String seg : s.split(String.valueOf(ch))) {
        res = Math.max(res, longestSubstring(seg, k));
      }
      return res;
    }
    // 原字符串没有小于 k 的字符串
    return s.length();
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

/** 子串相关，单词搜索参考 TTree */
class WWord extends DefaultSString {
  /**
   * 翻转字符串里的单词，对于 Java 不可能实现实际的 O(1) space，因此要求 s 原地即可
   *
   * <p>去空格 & 两次反转，全串与逐个单词
   *
   * @param s the s
   * @return string string
   */
  public String reverseWords(String s) {
    StringBuilder res = trimSpaces(s);
    reverse(res, 0, res.length() - 1);
    reverseEachWord(res);
    return res.toString();
  }

  // 去除字符首尾空格，并只保留单词间一个空格
  private StringBuilder trimSpaces(String s) {
    int lo = 0, hi = s.length() - 1;
    while (lo <= hi && s.charAt(lo) == ' ') {
      lo += 1;
    }
    while (lo <= hi && s.charAt(hi) == ' ') {
      hi -= 1;
    }
    StringBuilder res = new StringBuilder();
    while (lo <= hi) {
      char ch = s.charAt(lo);
      if (ch != ' ' || res.charAt(res.length() - 1) != ' ') {
        res.append(ch);
      }
      lo += 1;
    }
    return res;
  }

  // 循环至单词的末尾 & 翻转单词 & 步进
  private void reverseEachWord(StringBuilder sb) {
    int lo = 0, hi = 0;
    while (lo < sb.length()) {
      while (hi < sb.length() && sb.charAt(hi) != ' ') {
        hi += 1;
      }
      reverse(sb, lo, hi - 1);
      lo = hi + 1;
      hi += 1;
    }
  }

  /**
   * 单词拆分，s 能否被 wordDict 组合而成
   *
   * <p>dp[i] 表示 s 的前 i 位是否可以用 wordDict 中的单词表示，比如 wordDict=["apple", "pen", "code"]
   *
   * <p>则 s="applepencode" 有递推关系 dp[8]=dp[5]+check("pen")
   *
   * @param s the s
   * @param wordDict the word dict
   * @return boolean boolean
   */
  public boolean wordBreak(String s, List<String> wordDict) {
    boolean[] dp = new boolean[s.length() + 1];
    Map<String, Boolean> hash = new HashMap<>(wordDict.size());
    for (String word : wordDict) hash.put(word, true);
    dp[0] = true;
    for (int i = 1; i <= s.length(); i++) {
      // [0<-i] O(n^2)
      for (int j = i - 1; j >= 0; j--) {
        dp[i] = dp[j] && hash.getOrDefault(s.substring(j, i), false);
        if (dp[i]) break;
      }
    }
    return dp[s.length()];
  }

  /**
   * 比较版本号，逐个区间统计并比对
   *
   * @param version1 the version 1
   * @param version2 the version 2
   * @return int int
   */
  public int compareVersion(String version1, String version2) {
    int len1 = version1.length(), len2 = version2.length();
    int p1 = 0, p2 = 0;
    while (p1 < len1 || p2 < len2) {
      int n1 = 0, n2 = 0;
      while (p1 < len1 && version1.charAt(p1) != '.') {
        n1 = n1 * 10 + version1.charAt(p1) - '0';
        p1 += 1;
      }
      // 跳过点号
      p1 += 1;
      while (p2 < len2 && version2.charAt(p2) != '.') {
        n2 = n2 * 10 + version2.charAt(p2) - '0';
        p2 += 1;
      }
      p2 += 1;
      if (n1 != n2) {
        return n1 > n2 ? 1 : -1;
      }
    }
    return 0;
  }

  /**
   * 单词接龙，返回 beginWord 每次 diff 一个字母，最终变为 endWord 的最短路径，且所有路径均包含在 wordList 内
   *
   * <p>TODO 双向 bfs
   *
   * @param beginWord
   * @param endWord
   * @param wordList
   * @return
   */
  //  public int ladderLength(String beginWord, String endWord, List<String> wordList) {}
}

/**
 * 收集栈相关
 *
 * <p>关于 Java 模拟 stack 的选型
 * https://qastack.cn/programming/6163166/why-is-arraydeque-better-than-linkedlist
 */
class SStack {
  /**
   * 有效的括号
   *
   * <p>扩展1，需要保证优先级，如 {} 优先级最高即其 [{}] 非法，因此需要额外维护一个变量标识，在出入栈时更新
   *
   * <p>扩展2，左括号可不以正确的任意闭合，如 ([)] 返回true，同时不能视作同一种即只统计数量，如 {{][}}
   * 非法，即放弃对顺序的要求，而只要求同种的数量，因此使用三个变量统计数目而无需栈
   *
   * <p>扩展3，( & ) & * 三种符号，参下
   *
   * @param s the s
   * @return the boolean
   */
  public boolean isValid(String s) {
    // 外层括弧定义一个 Anonymous Inner Class
    // 内层括弧上是一个 instance initializer block，在内部匿名类构造时被执行
    Map<Character, Character> pairs =
        new HashMap<>(4) {
          {
            put('[', ']');
            put('(', ')');
            put('{', '}');
            put('?', '?');
          }
        };
    Deque<Character> stack = new ArrayDeque<>();
    // int level = 0;
    for (char ch : s.toCharArray()) {
      if (pairs.containsKey(ch)) {
        // if ((priorities.indexOf(ch) + 1) % 3 > level) return false;
        stack.add(ch);
        // level = Math.max((priorities.indexOf(ch) + 1) % 3, level);
        continue;
      }
      if (stack.size() == 0 || stack.getLast() == pairs.get(ch)) {
        return false;
      }
      // level = Math.max((priorities.indexOf(stack.get(stack.size() - 1)) + 1) % 3, level);
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
    return stack.size() == 0;
  }

  /**
   * 有效的括号字符串，贪心，参考
   * https://leetcode-cn.com/problems/valid-parenthesis-string/solution/you-xiao-de-gua-hao-zi-fu-chuan-by-leetc-osi3/
   *
   * <p>维护未匹配的左括号数量可能的上下界，未匹配的左括号数量必须非负
   *
   * <p>因此当最大值变成负数时，说明没有左括号可以和右括号匹配，返回
   *
   * <p>当最小值为 0 时，不应将最小值继续减少，以确保最小值非负
   *
   * <p>遍历结束时，所有的左括号都应和右括号匹配，因此只有当最小值为 0 时才满足
   *
   * @param s the s
   * @return boolean boolean
   */
  public boolean checkValidString(String s) {
    int minCount = 0, maxCount = 0;
    int n = s.length();
    for (int i = 0; i < n; i++) {
      char ch = s.charAt(i);
      if (ch == '(') {
        minCount += 1;
        maxCount += 1;
      } else if (ch == ')') {
        minCount = Math.max(minCount - 1, 0);
        maxCount -= 1;
        if (maxCount < 0) return false;
      } else if (ch == '*') {
        minCount = Math.max(minCount - 1, 0);
        maxCount += 1;
      }
    }
    return minCount == 0;
  }

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
   * 基本计算器 I & II 统一模板
   *
   * <p>TODO
   *
   * @param s the s
   * @return int int
   */
  //  public int calculate(String s) {
  //    return cal(s,0);
  //  }
  //  private int cal(String s, int begin) {
  //    // 当前一层，即同属一个括号内的数字组
  //    Deque<Integer> stack = new ArrayDeque<>();
  //    // 同一层上一个括号遭遇的运算符
  //    char preOper = '+';
  //    // 同一运算符的累计值
  //    int cur = 0;
  //    while (begin < s.length()) {
  //    char ch = s.charAt(begin);
  //    begin += 1;
  //    if (char == ' ') { // 空格跳过
  //      continue
  //    } else if '0' <= char && char <= '9' {
  //      // (c - '0')的这个括号不能省略，否则可能造成整型溢出
  //      // 因为c是一个 ASCII 码，如果不加括号就会先加后减，想象一下n如果接近 INT_MAX，就会溢出
  //      cur = cur*10 + int(char-'0')
  //    } else if char == '(' { // 左括号则递归处理
  //      cur = cal(s, begin)
  //    } else if char == ')' {
  //      break
  //    } else { // 还需要判断如果不是数字或空格，就把符号后边的数字处理后加入栈中，否则需要改动传入的s
  //      switch pre {
  //        case '+':
  //          stack = append(stack, cur)
  //        case '-':
  //          stack = append(stack, -cur)
  //        case '*':
  //          stack[len(stack)-1] *= cur
  //        default:
  //          stack[len(stack)-1] /= cur
  //      }
  //      pre = rune(char) // 更新当前符号 & 重置当前层
  //      cur = 0
  //    }
  //	}
  //    int res = 0;
  //    // 计算当前整一层的加和
  //    for (int num : stack) {
  //      res += num;
  //    }
  //    return res;
  //  }
}

/** 字符串遍历与解析相关 */
class PParse extends SString {
  /**
   * 字符串解码，类似压缩字符串 & 原子的数量 & 解码方法
   *
   * <p>需要分别保存计数和字符串，且需要两对分别保存当前和括号内
   *
   * <p>Integer.parseInt(c + "") 改为 c - '0'
   *
   * @param s the s
   * @return string string
   */
  public String decodeString(String s) {
    int curCount = 0;
    StringBuilder curStr = new StringBuilder();
    Deque<Integer> countStack = new ArrayDeque<>();
    Deque<String> strStack = new ArrayDeque<>();
    for (int i = 0; i < s.length(); i++) {
      char ch = s.charAt(i);
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
   * 字符串转换整数
   *
   * <p>去空格 & 特判空串 & 判断正负 & 逐位相加 & 判断溢出
   *
   * @param s the s
   * @return the int
   */
  public int myAtoi(String s) {
    int idx = 0, res = 0;
    boolean isNegative = false;
    while (idx < s.length() && s.charAt(idx) == ' ') {
      idx += 1;
    }
    if (idx == s.length()) {
      return 0;
    }
    if (s.charAt(idx) == '-') {
      idx += 1;
      isNegative = true;
    } else if (s.charAt(idx) == '+') {
      idx += 1;
    }
    while (idx < s.length()) {
      char ch = s.charAt(idx);
      if (ch < '0' || ch > '9') break;
      // 判断进位溢出
      int pre = res;
      res = res * 10 + (ch - '0');
      if (pre != res / 10) {
        return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
      }
      idx += 1;
    }
    return res * (isNegative ? -1 : 1);
  }

  /**
   * 压缩字符串
   *
   * <p>1.前指针遍历 & 找到同个字母的连续末尾并统计个数
   *
   * <p>2.后指针的下一位写入数量，注意数字需逆序写，并步进该指针
   *
   * @param chars the chars
   * @return int int
   */
  public int compress(char[] chars) {
    int lo = 0, len = 0;
    for (int hi = 0; hi < chars.length; hi++) {
      if (hi < chars.length - 1 && chars[hi] == chars[hi + 1]) {
        continue;
      }
      chars[lo] = chars[hi];
      lo += 1;
      int curLen = hi - len + 1;
      if (curLen > 1) {
        int start = lo;
        // 短除法，子串倒序写入原字符串，再将其反转
        while (curLen > 0) {
          chars[lo] = (char) (curLen % 10 + '0');
          lo += 1;
          curLen /= 10;
        }
        reverseChs(chars, start, lo - 1);
      }
      len = hi + 1;
    }
    return lo;
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
