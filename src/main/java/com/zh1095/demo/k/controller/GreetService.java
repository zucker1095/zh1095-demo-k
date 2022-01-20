package com.zh1095.demo.k.controller;

import com.zh1095.demo.k.model.api.greet.GreetRequest;
import com.zh1095.demo.k.model.api.greet.GreetResponse;

public interface GreetService {
  GreetResponse setup(GreetRequest greetRequest);
}
