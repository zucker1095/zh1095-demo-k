package com.zh1095.demo.k;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RequestMapping;

// HTTP server
@SpringBootApplication
public class Zh1095DemoKApplication {

  @RequestMapping("/ping")
  String ping() {
    return "pong";
  }

  public static void main(String[] args) {
    SpringApplication.run(Zh1095DemoKApplication.class, args);
  }
}
