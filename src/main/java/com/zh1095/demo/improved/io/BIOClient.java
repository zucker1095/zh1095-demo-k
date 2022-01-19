package com.zh1095.demo.improved.io;

import java.io.InputStream;
import java.net.ServerSocket;
import java.net.Socket;

public class BIOClient {
    public static void main(String[] args) throws Exception {
        ServerSocket serverSocket = new ServerSocket(6789);
        System.out.println("Server started...");
        while (true) {
            System.out.println("socket accepting...");
            Socket socket = serverSocket.accept();
            Runnable worker =
                    () -> {
                        try {
                            byte[] bytes = new byte[1024];
                            InputStream inputStream = socket.getInputStream();
                            while (true) {
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
                    };
            worker.run();
        }
    }
}
