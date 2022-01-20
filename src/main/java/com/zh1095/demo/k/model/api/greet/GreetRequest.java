package com.zh1095.demo.k.model.api.greet;

public class GreetRequest {
  private final String name;

  public GreetRequest(String name) {
    this.name = name;
  }

  public String getName() {
    return name;
  }
}
