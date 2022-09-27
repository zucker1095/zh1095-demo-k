package com.zh1095.demo.improved;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Main {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    String[] nums = in.nextLine().split(" ");
    int r = Integer.parseInt(nums[0]), c = Integer.parseInt(nums[1]);
    char[][] matrix = new char[r][c];
    for (int i = 0; i < r; i++) {
      String[] rows = in.nextLine().split("");
      for (int j = 0; j < c; j++) matrix[i][j] = rows[j].toCharArray()[0];
    }
    System.out.println(fm2(matrix));
  }

  public static int fm2(char[][] matrix) {
    int res = 0;
    DIR.put('^', new int[] {0, 1});
    DIR.put('v', new int[] {0, -1});
    DIR.put('<', new int[] {-1, 0});
    DIR.put('>', new int[] {1, 0});
    for (int i = 0; i < matrix.length; i++) {
      for (int j = 0; j < matrix[0].length; j++) {
        boolean[][] recStack = new boolean[matrix.length][matrix[0].length];
        res = Math.max(res, dfs(matrix, i, j, 1, recStack));
      }
    }
    return res;
  }

  private static final Map<Character, int[]> DIR = new HashMap<>(4);

  private static int dfs(char[][] matrix, int r, int c, int cur, boolean[][] recStack) {
    System.out.print(r);
    System.out.print(c);
    System.out.print('\n');
    if (!inArea(matrix, r, c)) return cur - 1;
    if (recStack[r][c]) return cur - 1;
    recStack[r][c] = true;
    int[] dir = DIR.get(matrix[r][c]);
    return dfs(matrix, r + dir[0], c + dir[1], cur + 1, recStack);
  }

  private static boolean inArea(char[][] matrix, int r, int c) {
    return r >= 0 && r <= matrix.length - 1 && c >= 0 && c <= matrix[0].length - 1;
  }

  private static final char[] TARGETS = "pony".toCharArray();

  public static int fm3(String s) {
    int res = 0;
    while (true) {
      int cur = 0, idx = 0;
      char[] chs = s.toCharArray();
      int[] paths = new int[4];
      while (idx < chs.length) {
        if (cur == 4) {
          res += 1;
          break;
        }
        if (chs[idx] == TARGETS[cur]) {
          paths[cur] = idx;
          cur += 1;
        }
        idx += 1;
      }
      if (idx == chs.length) break;
      if (cur == 4) s = gen(s, paths);
    }
    return res;
  }

  private static String gen(String s, int[] paths) {
    StringBuilder res = new StringBuilder(s.length() - 4);
    int idx = 0;
    for (int i = 0; i < s.length(); i++) {
      if (i == paths[idx]) {
        idx += 1;
        if (idx == 4) break;
        continue;
      }
      res.append(s.charAt(i));
    }
    return res.toString();
  }
}
