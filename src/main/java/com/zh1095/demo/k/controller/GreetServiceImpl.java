package com.zh1095.demo.k.controller;

import com.zh1095.demo.k.model.api.greet.GreetRequest;
import com.zh1095.demo.k.model.api.greet.GreetResponse;
import org.springframework.http.HttpStatus;

public class GreetServiceImpl implements GreetService {
  @Override
  public GreetResponse setup(GreetRequest req) {
    GreetResponse rsp = new GreetResponse();
    rsp.setCode(HttpStatus.OK);
    rsp.setMessage("hello" + req.getName());
    return rsp;
  }
}
