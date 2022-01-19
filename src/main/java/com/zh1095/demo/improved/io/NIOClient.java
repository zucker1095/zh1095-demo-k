package com.zh1095.demo.improved.io;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.Date;
import java.util.Scanner;

public class NIOClient {
  public static void main(String[] args) throws IOException {
    // 获取通道
    SocketChannel channel = SocketChannel.open(new InetSocketAddress("127.0.0.1", 6789));
    // 切换至非阻塞模式
    channel.configureBlocking(false);
    // 分配缓冲区大小
    ByteBuffer buffer = ByteBuffer.allocate(1024);
    Scanner scan = new Scanner(System.in);
    while (scan.hasNext()) {
      String next = scan.next();
      // 向缓冲区里写入数据
      buffer.put((new Date() + "\n" + next).getBytes());
      buffer.flip();
      // 向通道里写入带有数据的缓冲区对象， 表示向服务器发送数据
      channel.write(buffer);
      buffer.clear();
    }
    // 关闭通道
    channel.close();
  }
}
