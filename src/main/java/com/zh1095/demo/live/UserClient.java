package com.zh1095.demo.live;

import java.util.ArrayList;
import java.util.List;

public class UserClient {}

class User {
  private final int uid;
  private final List<Integer> friendUIDs = new ArrayList<>();

  // hot & active | follower & followee 我的雇员 & 我的雇主
  // private final List<Integer> followers = new ArrayList<>();
  // private final List<Integer> followees = new ArrayList<>();

  User(int uid) {
    this.uid = uid;
  }

  public int getUid() {
    return uid;
  }

  // 关注
  public void addFollower(int uid) {
    friendUIDs.add(uid);
  }

  // 取关
  public void removeFollower(int uid) {
    friendUIDs.remove(uid);
  }
}
