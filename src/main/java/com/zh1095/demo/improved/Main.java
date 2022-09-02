package com.zh1095.demo.improved;

import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Main {
  public static void main(String[] args) {
    Scanner in = new Scanner(System.in);
    int cnt = Integer.parseInt(in.nextLine());
    fm(new int[] {1, 2, 3}, new int[] {2, 1, 3});
  }

  static int[] postOrder;

  public static void fm(int[] preOrder, int[] inOrder) {
    postOrder = new int[inOrder.length];
    Map<Integer, Integer> inMap = new HashMap<>();
    for (int i = 0; i < inOrder.length; i++) {
      inMap.put(inOrder[i], i);
    }
    fm(preOrder, 0, inOrder.length - 1, inMap, 0);
    for (int n : postOrder) System.out.println(n);
  }

  public static void fm(
      int[] preOrder, int preLo, int preHi, Map<Integer, Integer> inMap, int inLo) {
    if (preLo < preOrder.length && preLo < preHi) {
      int val = preOrder[preLo], idx = inMap.get(val), cntL = idx - inLo;
      postOrder[preHi - cntL - 1] = val;
      fm(preOrder, preLo + 1, preLo + cntL, inMap, inLo);
      fm(preOrder, preLo + cntL + 1, preHi, inMap, inLo + 1);
    }
  }
}
