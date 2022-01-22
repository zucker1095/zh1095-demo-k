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
   * <p>扩展1，36 进制，转换为 10 进制
   *
   * <p>扩展2，相减，同理维护一个高位，负责减，注意前导零
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
    if (carry == 1) res.append(1);
    return res.reverse().toString();
  }

  private char getChar(int n) {
    return (char) n;
    // return n <= 9 ? (char) (n + '0') : (char) (n - 10 + 'a');
  }

  private int getInt(char ch) {
    return ('0' <= ch && ch <= '9') ? ch - '0' : ch - 'a' + 10;
  }

  /**
   * 字符串相乘，竖式
   *
   * @param num1 the num 1
   * @param num2 the num 2
   * @return string string
   */
  public String multiply(String num1, String num2) {
    if (num1.equals("0") || num2.equals("0")) return "0";
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
    StringBuilder result = new StringBuilder();
    for (int i = 0; i < res.length; i++) {
      if (i == 0 && res[i] == 0) continue; // 跳过前导零
      result.append(res[i]);
    }
    return result.toString();
  }

  /**
   * 最长公共前缀，纵向扫描
   *
   * @param strs the strs
   * @return string string
   */
  public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) return "";
    int count = strs.length;
    for (int i = 0; i < strs[0].length(); i++) {
      char pivot = strs[0].charAt(i);
      for (int j = 1; j < count; j++) {
        if (i == strs[j].length() || strs[j].charAt(i) != pivot) return strs[0].substring(0, i);
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
        start = i - ((len - 1) >> 1);
        end = i + (len >> 1);
      }
    }
    return s.substring(start, end + 1);
  }

  private int findLongestPalindrome(String s, int left, int right) {
    while (left > -1 && right < s.length() && s.charAt(left) == s.charAt(right)) {
      left -= 1;
      right += 1;
    }
    return right - left - 1;
  }
  /**
   * 第一个只出现一次的字符
   *
   * <p>扩展，第二个则添加一个计数器即可
   *
   * @param s the s
   * @return char char
   */
  public char firstUniqChar(String s) {
    // 只需要遍历一轮 s & hash，而 HashMap 需要两轮 s
    Map<Character, Boolean> hash = new LinkedHashMap<>();
    char[] sc = s.toCharArray();
    for (char c : sc) hash.put(c, !hash.containsKey(c));
    for (Map.Entry<Character, Boolean> d : hash.entrySet()) if (d.getValue()) return d.getKey();
    return ' ';
  }

  /**
   * 字符串解码，类似压缩字符串 & 原子的数量
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
    LinkedList<Integer> countStack = new LinkedList<>();
    LinkedList<String> strStack = new LinkedList<>();
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
   * <p>去空格 & 特判 & 判断正负 & 逐位相加 & 判断溢出
   *
   * @param s the s
   * @return the int
   */
  public int myAtoi(String s) {
    // 指针 & 符号
    int idx = 0, sign = 1;
    // last 记录上一次的 res 以判断溢出
    int res = 0, pre = 0;
    // 1.去空格
    while (idx < s.length() && s.charAt(idx) == ' ') {
      idx += 1;
    }
    // 特判全空串
    if (idx == s.length()) {
      return 0;
    }
    if (s.charAt(idx) == '-') {
      idx += 1;
      sign = -1;
    } else if (s.charAt(idx) == '+') {
      idx += 1;
    }
    while (idx < s.length()) {
      char ch = s.charAt(idx);
      if (ch < '0' || ch > '9') break;
      pre = res;
      res = res * 10 + ch - '0';
      if (pre != res / 10) // //如果不相等就是溢出了
      return (sign == (-1)) ? Integer.MIN_VALUE : Integer.MAX_VALUE;
      idx += 1;
    }
    return res * sign;
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
        // 为达到 O(1) space 需要自行实现将数字转化为字符串写入到原字符串的功能
        // 此处采用短除法将子串长度倒序写入原字符串中，然后再将其反转即可
        while (curLen > 0) {
          chars[lo] = (char) (curLen % 10 + '0');
          lo += 1;
          curLen /= 10;
        }
        reverseString(chars, start, lo - 1);
      }
      len = hi + 1;
    }
    return lo;
  }
}

/** 子串相关，单词搜索参考 TTree */
class WWord extends DefaultSString {
  /**
   * 反转字符串
   *
   * @param s the s
   */
  public void reverseString(char[] s) {
    reverseString(s, 0, s.length - 1);
  }

  /**
   * 翻转字符串里的单词，对于 Java 不可能实现实际的 O(1) space，因此要求 s 原地即可
   *
   * <p>去空格 & 两次反转
   *
   * @param s the s
   * @return string string
   */
  public String reverseWords(String s) {
    StringBuilder sb = trimSpaces(s);
    reverse(sb, 0, sb.length() - 1);
    reverseEachWord(sb);
    return sb.toString();
  }

  // 1.定位字符串首个非空的字符
  // 2.逆向定位字符串首个非空的字符
  // 3.字符串间的空白字符只保留一个
  private StringBuilder trimSpaces(String s) {
    int lo = 0, hi = s.length() - 1;
    while (lo <= hi && s.charAt(lo) == ' ') lo += 1;
    while (lo <= hi && s.charAt(hi) == ' ') hi -= 1;
    StringBuilder sb = new StringBuilder();
    while (lo <= hi) {
      char c = s.charAt(lo);
      if (c != ' ' || sb.charAt(sb.length() - 1) != ' ') sb.append(c);
      lo += 1;
    }
    return sb;
  }

  // 循环至单词的末尾 & 翻转单词 & 步进
  private void reverseEachWord(StringBuilder sb) {
    int lo = 0, hi = 0;
    while (lo < sb.length()) {
      while (hi < sb.length() && sb.charAt(hi) != ' ') hi += 1;
      reverse(sb, lo, hi - 1);
      lo = hi + 1;
      hi += 1;
    }
  }

  /**
   * 单词拆分
   *
   * <p>dp[i] 表示 s 的前 i 位是否可以用 wordDict 中的单词表示，比如 wordDict=["apple", "pen", "code"]
   *
   * <p>s="applepencode" 则 dp[8] = dp[5] + check("pen")
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
      for (int j = i - 1; j >= 0; j--) { // [0<-i] O(n^2)
        dp[i] = dp[j] && hash.getOrDefault(s.substring(j, i), false);
        if (dp[i]) break;
      }
    }
    return dp[s.length()];
  }

  /**
   * 比较版本号
   *
   * @param version1 the version 1
   * @param version2 the version 2
   * @return int int
   */
  public int compareVersion(String version1, String version2) {
    int n = version1.length(), m = version2.length();
    int p1 = 0, p2 = 0;
    while (p1 < n || p2 < m) {
      int num1 = 0, num2 = 0; // 逐个区间计算
      while (p1 < n && version1.charAt(p1) != '.') {
        num1 = num1 * 10 + version1.charAt(p1) - '0';
        p1 += 1;
      }
      p1 += 1; // 跳过点号
      while (p2 < m && version2.charAt(p2) != '.') {
        num2 = num2 * 10 + version2.charAt(p2) - '0';
        p2 += 1;
      }
      p2 += 1; // 同上
      if (num1 != num2) return num1 > num2 ? 1 : -1;
    }
    return 0;
  }
}

/** 滑动窗口相关 */
class WWindow {
  /**
   * 无重复字符的最长子串，sliding window
   *
   * @param s the s
   * @return the int
   */
  public int lengthOfLongestSubstring(String s) {
    if (s.length() == 0) return 0;
    Map<Character, Integer> window = new HashMap<>();
    int max = 0, lo = 0;
    for (int hi = 0; hi < s.length(); hi++) {
      if (window.containsKey(s.charAt(hi))) lo = Math.max(lo, window.get(s.charAt(hi)) + 1);
      window.put(s.charAt(hi), hi);
      max = Math.max(max, hi - lo + 1);
    }
    return max;
  }

  /**
   * 最小覆盖字串，sliding window
   *
   * @param s the s
   * @param t the t
   * @return string string
   */
  public String minWindow(String s, String t) {
    Map<Character, Integer> need = new HashMap<>();
    for (char c : s.toCharArray()) need.put(c, 0);
    for (char c : t.toCharArray()) {
      if (need.containsKey(c)) need.put(c, need.get(c) + 1);
      else return "";
    }
    String res = "";
    int length = Integer.MAX_VALUE, counter = t.length();
    int lo = 0, hi = 0;
    while (hi < s.length()) {
      char add = s.charAt(hi);
      hi += 1;
      if (need.get(add) > 0) counter -= 1;
      need.put(add, need.get(add) - 1);
      while (counter == 0) {
        if (length > hi - lo) {
          length = hi - lo;
          res = s.substring(lo, hi);
        }
        char out = s.charAt(lo);
        lo += 1;
        if (need.get(out) == 0) counter += 1;
        need.put(out, need.get(out) + 1);
      }
    }
    return res;
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
    MonotonicQueue mq = new MonotonicQueue();
    for (int i = 0; i < nums.length; i++) {
      mq.push(nums[i]);
      if (i < k - 1) continue;
      res[i - k + 1] = mq.max();
      mq.pop(nums[i - k + 1]);
    }
    return res;
  }
  /** The type Monotonic queue. */
  private static class MonotonicQueue {
    private final Deque<Integer> monotonicQueue = new LinkedList<>();

    /**
     * Push.
     *
     * @param num the num
     */
    public void push(int num) {
      while (monotonicQueue.size() > 0 && monotonicQueue.getLast() < num)
        monotonicQueue.removeLast();
      monotonicQueue.addLast(num);
    }

    /**
     * Pop.
     *
     * @param num the num
     */
    public void pop(int num) {
      if (monotonicQueue.size() > 0 && monotonicQueue.getFirst() == num)
        monotonicQueue.removeFirst();
    }

    /**
     * Max int.
     *
     * @return the int
     */
    public int max() {
      return monotonicQueue.getFirst();
    }
  }
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
   * <p>扩展，需要保证优先级，如 {} 优先级最高即其 [{}] 非法，因此需要额外维护一个变量标识，在出入栈时更新
   *
   * @param s the s
   * @return the boolean
   */
  public boolean isValid(String s) {
    //    ArrayList<Character> priorities =
    //        new ArrayList<Character>() {
    //          {
    //            add('(');
    //            add(')');
    //            add('[');
    //            add(']');
    //            add('{');
    //            add('}');
    //          }
    //        };
    // 第一层括弧定义了一个 Anonymous Inner Class
    // 第二层括弧上是一个 instance initializer block，在内部匿名类构造时被执行
    Map<Character, Character> pairs =
        new HashMap<Character, Character>(4) {
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
      if (stack.size() == 0 || stack.getLast() == pairs.get(ch)) return false;
      // level = Math.max((priorities.indexOf(stack.get(stack.size() - 1)) + 1) % 3, level);
      stack.removeLast();
    }
    return stack.size() == 0;
  }

  /**
   * 删除字符串中的所有相邻重复项，不保留，且需要反复执行
   *
   * @param s the s
   * @return string string
   */
  public String removeDuplicates(String s) {
    StringBuilder stack = new StringBuilder();
    int top = -1;
    for (int i = 0; i < s.length(); i++) {
      char cur = s.charAt(i);
      if (top >= 0 && stack.charAt(top) == cur) {
        stack.deleteCharAt(top);
        top -= 1;
      } else {
        stack.append(cur);
        top += 1;
      }
    }
    return stack.toString();
  }

  /**
   * 有效的括号字符串，左加右减星减加
   *
   * @param s the s
   * @return boolean boolean
   */
  public boolean checkValidString(String s) {
    // 令左括号的得分为 1，右为 −1，那么最终得分需为 0，由于存在 *，因此遍历 ing 只能估计最终值，即其区间
    int minCount = 0, maxCount = 0;
    int n = s.length();
    for (int i = 0; i < n; i++) {
      char c = s.charAt(i);
      if (c == '(') {
        minCount += 1;
        maxCount += 1;
      } else if (c == ')') {
        // 当最小值为 0 时，不应将其继续减少，以确保其非负
        minCount = Math.max(minCount - 1, 0);
        maxCount -= 1;
        // 未匹配的左括号数量必须非负，因此当最大值变成负数时，说明没有左括号可以和右括号匹配，返回
        if (maxCount < 0) return false;
      } else if (c == '*') {
        minCount = Math.max(minCount - 1, 0);
        maxCount += 1;
      }
    }
    return minCount == 0;
  }

  /**
   * 每日温度，单调栈，递减
   *
   * @param temperatures the t
   * @return int [ ]
   */
  public int[] dailyTemperatures(int[] temperatures) {
    Deque<Integer> stack = new ArrayDeque<>();
    int[] res = new int[temperatures.length];
    for (int i = 0; i < temperatures.length; i++) {
      while (!stack.isEmpty() && temperatures[i] > temperatures[stack.getLast()]) {
        int pre = stack.removeLast();
        res[pre] = i - pre;
      }
      stack.addLast(i);
    }
    return res;
  }

  /**
   * 最小栈
   *
   * @author cenghui
   */
  public class MinStack {
    private final Deque<Integer> stack = new ArrayDeque<>();
    private int min;

    /**
     * Push.
     *
     * @param x the x
     */
    public void push(int x) {
      if (stack.isEmpty()) min = x;
      stack.push(x - min); // 存差
      if (x < min) min = x; // 更新
    }

    /** Pop. */
    public void pop() {
      if (stack.isEmpty()) return;
      int pop = stack.pop();
      // 弹出的是负值，要更新 min
      if (pop < 0) min -= pop;
    }

    /**
     * Top int.
     *
     * @return the int
     */
    public int top() {
      int top = stack.peek();
      // 负数的话，出栈的值保存在 min 中，出栈元素加上最小值即可
      return top < 0 ? min : top + min;
    }

    /**
     * Gets min.
     *
     * @return the min
     */
    public int getMin() {
      return min;
    }
  }
  /**
   * 基本计算器 I & II 统一模板
   *
   * @param s the s
   * @return int int
   */
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
  protected void reverseString(char[] chars, int left, int right) {
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
