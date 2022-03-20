package com.zh1095.demo.improved.algorithmn;

import java.util.*;

/**
 * 收集字符串相关，个人遵从如下原则，如果部分查找不到，则至 DDP
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
  private final String HEX_CHAR = "0123456789abcdef";

  /**
   * 字符串相加，双指针同时遍历 & 比对 & 最后处理高位，模板保持 mergeTwoLists & addStrings & addTwoNumbers 一致
   *
   * <p>类似包括合并 & 两数相加 & 大数相乘 & 大数相减 & 36 进制
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
      res.append(getChar(tmp % base));
      p1 -= 1;
      p2 -= 1;
    }
    return res.reverse().toString();
  }

  private char getChar(int num) {
    return num <= 9 ? (char) (num + '0') : (char) (num - 10 + 'a');
  }

  // ASCII 编码允许直接相减
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
    // 特判
    if (num1.equals("0") || num2.equals("0")) {
      return "0";
    }
    // 乘法比加法多出高位的概率更大，因此额外冗余一位暂存计算结果即可，取缔 carry
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
   * 字符串转换整数，如 " -26" to 26
   *
   * <p>去空格 & 判正负 & 逐位加 & 判溢出
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
    if (idx == s.length()) return 0;
    if (s.charAt(idx) == '-') isNegative = true;
    if (s.charAt(idx) == '-' || s.charAt(idx) == '+') idx += 1;
    for (int i = idx; i < s.length(); i++) {
      char ch = s.charAt(i);
      if (ch < '0' || ch > '9') break;
      int pre = res;
      res = res * 10 + (ch - '0');
      if (pre != res / 10) {
        return isNegative ? Integer.MIN_VALUE : Integer.MAX_VALUE;
      }
    }
    return res * (isNegative ? -1 : 1);
  }

  /**
   * 压缩字符串，原地字符串编码，如 [a,a,a,b,b] to [a,3,b,2]，前者即 "aaabb" 后者同理 "a3b2"
   *
   * <p>框架类似「移动零」与滑窗，前后指针分别作读写 & 统计重复区间 & 写入
   *
   * @param chars the chars
   * @return int int
   */
  public int compress(char[] chars) {
    // write & read
    int lo = 0, hi = 0;
    while (hi < chars.length) {
      int cur = hi;
      while (cur < chars.length && chars[hi] == chars[cur]) {
        cur += 1;
      }
      chars[lo] = chars[hi];
      lo += 1;
      // 多位数字逐位写入
      if (cur - hi > 1) {
        for (char digit : Integer.toString(cur - hi).toCharArray()) {
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
    long cur = num < 0 ? (long) (Math.pow(2, 32) + num) : num;
    StringBuilder res = new StringBuilder();
    while (cur != 0) {
      res.append(HEX_CHAR.charAt((int) (cur % 16)));
      cur /= 16;
    }
    return res.reverse().toString();
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

  /**
   * 比较版本号，逐个区间计数
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
      if (n1 != n2) return n1 > n2 ? 1 : -1;
    }
    return 0;
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
    // ASCII 表总长 128
    int[] window = new int[128];
    int res = 1;
    int lo = 0, hi = 0;
    while (hi < s.length()) {
      char in = s.charAt(hi);
      window[in] += 1;
      while (window[in] == 2) {
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
    // 遍历的指针与结果的始末
    int lo = 0, hi = 0, start = 0, end = Integer.MAX_VALUE;
    int counter = t.length();
    int[] needle = new int[128];
    for (char ch : t.toCharArray()) {
      needle[ch] += 1;
    }
    while (hi < s.length()) {
      char in = s.charAt(hi);
      if (needle[in] > 0) counter -= 1;
      needle[in] -= 1;
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
   * 滑动窗口的最大值，offer & max & poll
   *
   * @param nums the nums
   * @param k the k
   * @return int [ ]
   */
  public int[] maxSlidingWindow(int[] nums, int k) {
    int[] segmentMax = new int[nums.length - k + 1];
    Deque<Integer> monotonicQueue = new LinkedList<>();
    for (int i = 0; i < nums.length; i++) {
      // offer
      while (monotonicQueue.size() > 0 && monotonicQueue.peekLast() < nums[i]) {
        monotonicQueue.pollLast();
      }
      monotonicQueue.offerLast(nums[i]);
      if (i < k - 1) continue;
      int outIdx = i - k + 1;
      //      res[outIdx] = mq.max();
      segmentMax[outIdx] = monotonicQueue.peekFirst();
      //      mq.poll(nums[outIdx]);
      if (monotonicQueue.size() > 0 && monotonicQueue.peekFirst() == nums[outIdx]) {
        monotonicQueue.pollFirst();
      }
    }
    return segmentMax;
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

/**
 * 收集栈相关
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
      // 左括号入栈
      if (!pairs.containsKey(ch)) {
        // if ((priorities.indexOf(ch) + 1) % 3 > level) return false;
        stack.addLast(ch);
        // level = Math.max((priorities.indexOf(ch) + 1) % 3, level);
        continue;
      }
      // 右括号出栈并判断
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
   * <p>1.下界至少为0
   *
   * <p>2.上界不能为负
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
   * 字符串解码，如 3[a]2[bc] to aaabcbc，类似压缩字符串 & 原子的数量
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
   * 简化路径
   *
   * @param path
   * @return
   */
  public String simplifyPath(String path) {
    Deque<String> stack = new ArrayDeque<>();
    for (String segment : path.split("/")) {
      if (!stack.isEmpty() && segment.equals("..")) stack.pollLast();
      else if (!" ..".contains(segment)) stack.offerLast(segment);
    }
    StringBuilder res = new StringBuilder();
    for (String i : stack) {
      res.append("/" + i);
    }
    return res.length() == 0 ? "/" : res.toString();
  }

  /**
   * 判断两个字符串是否含义一致，二者只包含 (,),+,-,a-z 且保证字母不会连续，即合法的多元一次表达式
   *
   * <p>只有加减与括号，则展开括号并单栈暂存运算符即可，代码模板参下「字符串解码」
   *
   * <p>思路参考「基本计算器」https://leetcode-cn.com/problems/basic-calculator/solution/ji-ben-ji-suan-qi-by-wo-yao-chu-qu-luan-nae94/
   *
   * @return
   */
  public boolean isSame(String s1, String s2) {
    int[] count1 = countLetter(s1), count2 = countLetter(s2);
    for (int i = 0; i < 26; i++) {
      if (count1[i] != count2[i]) return false;
    }
    return true;
  }

  private int[] countLetter(String s) {
    int[] curCounter = new int[26];
    int curSign = 1;
    Deque<Integer> opStack = new ArrayDeque<>();
    opStack.addLast(1);
    for (char ch : s.toCharArray()) {
      ch = Character.toLowerCase(ch);
      if ('a' <= ch && ch <= 'z') curCounter[ch - 'a'] += curSign == 1 ? 1 : -1;
      else if (ch == '+') curSign = opStack.getLast();
      else if (ch == '-') curSign = -opStack.getLast();
      else if (ch == '(') opStack.addLast(curSign);
      else if (ch == ')') opStack.removeLast();
    }
    return curCounter;
  }

  /**
   * 基本计算器
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
    //    s = s.replaceAll(" ", "");
    // 所有的数字
    Deque<Integer> numStack = new ArrayDeque<>();
    // 防止第一个数为负数
    numStack.addLast(0);
    // 存放所有非数字的操作
    Deque<Character> opStack = new ArrayDeque<>();
    char[] chs = s.toCharArray();
    for (int i = 0; i < chs.length; i++) {
      char cur = chs[i];
      if (cur == ' ') continue;
      if (cur == '(') {
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
    // 需要特判
    if (strs.length == 0) return "";
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
    int lo = 0, hi = 0;
    for (int i = 0; i < s.length(); i++) {
      int odd = findLongestPalindrome(s, i, i), even = findLongestPalindrome(s, i, i + 1);
      int len = Math.max(odd, even);
      if (len > hi - lo) {
        lo = i - (len - 1) / 2;
        hi = i + len / 2;
      }
    }
    return s.substring(lo, hi + 1);
  }

  // 分别从 lo & hi 扩散，直到二者所在字符不同
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
   * 实现strStr()，返回 haystack 中匹配 needle 的首个子串的首个字符的索引
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/implement-strstr/solution/shua-chuan-lc-shuang-bai-po-su-jie-fa-km-tb86/
   *
   * @param haystack
   * @param needle
   * @return
   */
  public int strStr(String haystack, String needle) {
    if (needle.isEmpty()) return 0;
    // 分别读取原串和匹配串的长度
    int lenH = haystack.length(), lenN = needle.length();
    // 原串和匹配串前面都加空格，使其下标从 1 开始
    haystack = " " + haystack;
    needle = " " + needle;
    char[] hayChs = haystack.toCharArray(), needChs = needle.toCharArray();
    // 构建 next 数组，和匹配串相关
    int[] next = new int[lenN + 1];
    // 构造过程 i=2 j=0 开始，i 小于等于匹配串长度
    for (int i = 2, j = 0; i <= lenN; i++) {
      // 匹配不成功的话，j = next(j)
      while (j > 0 && needChs[i] != needChs[j + 1]) {
        j = next[j];
      }
      // 匹配成功的话，先让 j++
      if (needChs[i] == needChs[j + 1]) j += 1;
      // 更新 next[i]，结束本次循环，i++
      next[i] = j;
    }
    // 匹配过程，i=1 j=0 开始，i 小于等于原串长度
    for (int i = 1, j = 0; i <= lenH; i++) {
      // 匹配不成功 j = next(j)
      while (j > 0 && hayChs[i] != needChs[j + 1]) {
        j = next[j];
      }
      // 匹配成功的话，先让 j++，结束本次循环后 i++
      if (hayChs[i] == needChs[j + 1]) j += 1;
      // 整一段匹配成功，直接返回下标
      if (j == lenN) return i - lenN;
    }
    return -1;
  }

  /**
   * 判断子序列，顺序满足，因此双指针正序遍历即可
   *
   * <p>扩展1，依次检查海量 s 是否均为 t 的子序列，参下
   * https://leetcode-cn.com/problems/is-subsequence/solution/dui-hou-xu-tiao-zhan-de-yi-xie-si-kao-ru-he-kuai-s/
   *
   * @param s pattern
   * @param t main
   * @return boolean boolean
   */
  public boolean isSubsequence(String s, String t) {
    return isSubsequence1(s, t);
  }

  private boolean isSubsequence1(String s, String t) {
    int p1 = 0, p2 = 0;
    while (p1 < s.length() && p2 < t.length()) {
      if (s.charAt(p1) == t.charAt(p2)) p1 += 1;
      p2 += 1;
    }
    return p1 == s.length();
  }

  // TODO KMP 思想，类似于用伪链表把相同的字符给链接起来
  // dp[i][p] 表示 patternStr[i+1:end] 区间内 p 的首个索引
  // 在 dp 的定义下，如对于 "abac" 无法找到首个 a，因此需要前缀一个空格
  // 该解法中对 patternStr 的处理与 mainStr 无关，且预处理完成后，可以利用预处理数组的信息，线性地算出任意一个字符串 mainStr 是否为 patternStr 的子串
  private boolean isSubsequence2(String mainStr, String patternStr) {
    patternStr = " " + patternStr;
    int[][] dp = new int[patternStr.length()][26];
    for (int ch = 0; ch < 26; ch++) {
      int nxt = -1;
      for (int i = patternStr.length() - 1; i >= 0; i--) {
        dp[i][ch] = nxt;
        if (patternStr.charAt(i) == ch + 'a') nxt = i;
      }
    }
    // 起始位置是空字符
    int idx = 0;
    for (char ch : mainStr.toCharArray()) {
      idx = dp[idx][ch - 'a'];
      if (idx == -1) return false;
    }
    return true;
  }
}

/** 子串相关，单词搜索参考 TTree */
class WWord extends DefaultSString {
  /**
   * 翻转字符串里的单词，对于 Java 不可能实现实际的 O(1) space，因此要求 s 原地即可
   *
   * <p>去空格 & 整串翻转 & 逐个翻转
   *
   * @param s the s
   * @return string string
   */
  public String reverseWords(String s) {
    StringBuilder res = trimSpaces(s);
    reverse(res, 0, res.length() - 1);
    int lo = 0, hi = 0;
    // 循环至单词的末尾 & 翻转单词 & 步进
    while (lo < res.length()) {
      while (hi < res.length() && res.charAt(hi) != ' ') {
        hi += 1;
      }
      reverse(res, lo, hi - 1);
      lo = hi + 1;
      hi += 1;
    }
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
   * 单词接龙，返回 beginWord 每次 diff 一个字母，最终变为 endWord 的最短路径，且所有路径均包含在 wordList 内
   *
   * <p>TODO 参考
   * https://leetcode-cn.com/problems/word-ladder/solution/yan-du-you-xian-bian-li-shuang-xiang-yan-du-you-2/
   *
   * @param beginWord
   * @param endWord
   * @param wordList
   * @return
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
