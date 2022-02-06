package com.zh1095.demo.improved.grammar;

import java.util.Scanner;

public class SScanner {
  public static void main(String[] args) {
    in();
  }

  // digit & letter & array
  private static void in() {
    Scanner scanner = new Scanner(System.in); // 创建Scanner对象
    System.out.print("digit: "); // 打印提示
    int digit = scanner.nextInt(); // 读取一行输入并获取整数
    System.out.print("letter: "); // 打印提示
    String letter = scanner.nextLine(); // 读取一行输入并获取字符串
    System.out.print("array: "); // 格式化输出
    String arr = scanner.nextLine();
    String[] _arr = arr.substring(1, arr.length() - 1).split(",");
  }

  private static void out() {}
}
