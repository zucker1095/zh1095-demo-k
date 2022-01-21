package com.zh1095.demo.k.clients.zh1095_demo;

import com.zh1095.demo.k.controller.GreetService;
import com.zh1095.demo.k.model.api.greet.GreetRequest;
import com.zh1095.demo.k.model.api.greet.GreetResponse;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.ReferenceConfig;
import org.apache.dubbo.config.RegistryConfig;

// 作为 client 测试 zh1095.demo.j
public class j {
  private static final String zookeeperHost = System.getProperty("zookeeper.address", "127.0.0.1");
  private static final String PSM = "zh1095.demo.j";

  public static void main(String[] args) {
    ReferenceConfig<GreetService> reference = new ReferenceConfig<>();
    reference.setApplication(new ApplicationConfig(PSM));
    reference.setRegistry(new RegistryConfig("zookeeper://" + zookeeperHost + ":2181"));
    reference.setInterface(GreetService.class);
    GreetService service = reference.get();
    GreetResponse rsp = service.setup(new GreetRequest("dubbo"));
    System.out.println(rsp.getMessage());
  }
}
