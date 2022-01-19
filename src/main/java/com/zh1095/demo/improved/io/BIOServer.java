package com.zh1095.demo.improved.io;

import java.io.IOException;
import java.io.InputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class BIOServer {
    public static void main(String[] args) throws IOException {
        // 快速
        ExecutorService cached = Executors.newCachedThreadPool();
        // 创建 ServerSocket
        ServerSocket serverSocket = new ServerSocket(6789);
        System.out.println("Server started....");
        while (true) {
            printThreadMsg();
            // listen, wait for client
            System.out.println("Waiting for connect...");
            // connect
            final Socket socket = serverSocket.accept();
            System.out.println("Client connect...");
            cached.execute(
                    () -> {
                        handle(socket);
                    });
        }
    }

    private static void handle(Socket socket) {
        try {
            printThreadMsg();
            byte[] bytes = new byte[1024];
            // get InputStream by socket
            InputStream inputStream = socket.getInputStream();
            while (true) {
                printThreadMsg();
                System.out.println("reading...");
                int read = inputStream.read(bytes);
                if (read != -1) System.out.println(new String(bytes, 0, read));
                else break;
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                socket.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private static void printThreadMsg() {
        System.out.println(
                "Thread's id = "
                        + Thread.currentThread().getId()
                        + " Thread's name = "
                        + Thread.currentThread().getName());
    }
}
