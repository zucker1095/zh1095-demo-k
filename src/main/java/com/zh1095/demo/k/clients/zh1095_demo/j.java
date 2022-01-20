package com.zh1095.demo.k.clients.zh1095_demo;

import com.zh1095.demo.k.controller.GreetService;
import com.zh1095.demo.k.model.api.greet.GreetRequest;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.ReferenceConfig;
import org.apache.dubbo.config.RegistryConfig;

public class j {
  private static String zookeeperHost = System.getProperty("zookeeper.address", "127.0.0.1");

  public static void main(String[] args) {
    ReferenceConfig<GreetService> reference = new ReferenceConfig<>();
    reference.setApplication(new ApplicationConfig("first-dubbo-consumer"));
    reference.setRegistry(new RegistryConfig("zookeeper://" + zookeeperHost + ":2181"));
    reference.setInterface(GreetService.class);
    GreetService service = reference.get();
    String message = service.setup(new GreetRequest("dubbo"));
    System.out.println(message);
  }
}
