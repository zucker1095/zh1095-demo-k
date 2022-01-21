package com.zh1095.demo.k.rpc;

import com.zh1095.demo.k.controller.GreetService;
import com.zh1095.demo.k.controller.GreetServiceImpl;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.RegistryConfig;
import org.apache.dubbo.config.ServiceConfig;
import java.util.concurrent.CountDownLatch;

// HTTP server 嵌入 RPC server
public class Application {
  public static void main(String[] args) throws Exception {
    registerGreet();
    //    registerUserInfo();
    System.out.println("dubbo service started");
    new CountDownLatch(1).await();
  }

  private static final String PSM = "zh1095.demo.j";
  private static final String ZookeeperHost = System.getProperty("zookeeper.address", "127.0.0.1");
  private static final int RPC_PORT = 2181;

  private static void setupService(ServiceConfig<?> service) {
    service.setApplication(new ApplicationConfig(PSM));
    service.setRegistry(new RegistryConfig("zookeeper://" + ZookeeperHost + ":" + RPC_PORT));
  }

  private static void registerGreet() {
    ServiceConfig<GreetService> service = new ServiceConfig<>();
    setupService(service);
    service.setInterface(GreetService.class);
    service.setRef(new GreetServiceImpl());
    service.export();
  }

  private static void registerUserInfo() {
    ServiceConfig<GreetService> service = new ServiceConfig<>();
    setupService(service);
    service.setInterface(GreetService.class);
    service.setRef(new GreetServiceImpl());
    service.export();
  }
}
