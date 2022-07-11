package com.zh1095.demo.improved;

import java.util.Scanner;

public class Main {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    String input = in.nextLine(); // [1,2,3,4]
    int[] arr = new int[input.length() - 2];
    for (String seg : input.substring(1, input.length() - 2).split(",")) {}
    System.out.println(arr);
    // 回车后
    int cnt = Integer.parseInt(in.nextLine()); // 2
    System.out.println(cnt);
  }
}
