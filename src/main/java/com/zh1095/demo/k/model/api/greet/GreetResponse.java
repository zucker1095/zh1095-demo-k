package com.zh1095.demo.k.model.api.greet;

import org.springframework.http.HttpStatus;

public class GreetResponse {
  private HttpStatus code;
  private String message;

  public HttpStatus getCode() {
    return code;
  }

  public void setCode(HttpStatus code) {
    this.code = code;
  }

  public String getMessage() {
    return message;
  }

  public void setMessage(String message) {
    this.message = message;
  }
}
