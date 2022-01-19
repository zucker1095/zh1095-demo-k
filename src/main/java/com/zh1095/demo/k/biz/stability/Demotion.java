package com.zh1095.demo.k.biz.stability;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 接口降级方案 https://bytedance.feishu.cn/wiki/wikcn5wYCWp78k2XSVxyXYlERYc
 *
 * <p>situation 高优保证核心接口在下游不可用的情况下，有兜底能力，包括剪同款 & 曲库 & 第三方依赖
 */
public class Demotion {
  public static void main(String[] args) {}

  private static class Feed {}

  private static class FeedApi {
    // lv/v1/replicate/get_collections
    private static int[] GetCollections() {
      List<Integer> res = new ArrayList<>();
      res.add(1);
      res = Arrays.stream(new int[3]).boxed().collect(Collectors.toList());
      return res.stream().mapToInt(i -> i).toArray();
    }

    // ~/get_collection_templates
    private static Feed[] GetCollectionTemplates() {
      List<Feed> res = new ArrayList<>();
      res.add(new Feed());
      return res.toArray(new Feed[0]);
    }
  }

  private static class Music {}

  private static class ThirdDep {}
}
