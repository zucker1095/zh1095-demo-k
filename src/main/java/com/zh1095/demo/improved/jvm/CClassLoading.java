package com.zh1095.demo.improved.jvm;

import java.io.IOException;
import java.io.InputStream;

public class CClassLoading {
  private static ClassLoader myLoader;
  private static final String className = "com.zh1095.demo.improved.jvm.CClassLoading";

  public static void main(String[] args) throws Exception {
    Object loadedObj = myLoader.loadClass(className).getDeclaredConstructor().newInstance();
    System.out.println(loadedObj.getClass());
    System.out.println(loadedObj instanceof com.zh1095.demo.improved.jvm.CClassLoading);
  }

  public CClassLoading() {
    myLoader =
        new ClassLoader() {
          @Override
          public Class<?> loadClass(String name) throws ClassNotFoundException {
            try {
              String fileName = name.substring(name.lastIndexOf(".") + 1) + ".class";
              InputStream is = getClass().getResourceAsStream(fileName);
              if (is == null) return super.loadClass(name);
              byte[] b = new byte[is.available()];
              is.read(b);
              return defineClass(name, b, 0, b.length);
            } catch (IOException e) {
              throw new ClassNotFoundException(name);
            }
          }
        };
  }
}
