package com.zh1095.demo.improved.algorithmn;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/** 收集所有场景设计 & 系统设计类 */
public class Design {}

/** 设计 C 内存分配与释放 */
class MMemory {
  public void malloc() {}

  public void free() {}
}

/** 压缩类 */
class Compress {
  /**
   * 字典树，适用于动态添加
   *
   * @author cenghui
   */
  public class Trie {
    private final int ALPHABET_SIZE = 26;
    private TrieNode root;

    /**
     * 如果不存在，则插入，如果属于某个的前缀，则标记为叶
     *
     * @param key the key
     */
    public void insert(String key) {
      int level, idx; // 当前层级 & 同一层的索引
      TrieNode cur = root; // 当前节点
      for (level = 0; level < key.length(); level++) {
        idx = key.charAt(level) - 'a'; // offset
        if (cur.children[idx] == null) cur.children[idx] = new TrieNode();
        cur = cur.children[idx];
      }
      // mark last node as leaf
      cur.isEndOfWord = true;
    }

    /**
     * Returns true if key presents in trie, else false
     *
     * @param key the key
     * @return the boolean
     */
    public boolean search(String key) {
      int level, idx;
      TrieNode cur = root;
      for (level = 0; level < key.length(); level++) {
        idx = key.charAt(level) - 'a';
        if (cur.children[idx] == null) return false;
        cur = cur.children[idx];
      }
      return cur.isEndOfWord;
    }

    private class TrieNode {
      private final TrieNode[] children = new TrieNode[ALPHABET_SIZE];
      private boolean isEndOfWord = false; // 默认
      // isEndOfWord = false;
      // for (int i = 0; i < ALPHABET_SIZE; i++) children[i] = null;
    }
  }

  /**
   * 位图
   *
   * @author cenghui
   */
  public class BitMap {
    private final byte[] bits;
    /**
     * Instantiates a new Bit map.
     *
     * <p>1bit can store 8 data, so how many bits are needed for capacity data, so capacity/8+1
     * right shifting 3 bits is equivalent to dividing by 8.
     *
     * @param capacity the capacity
     */
    public BitMap(int capacity) {
      bits = new byte[(capacity >> 3) + 1];
    }

    /**
     * Add.
     *
     * <p>After moving 1 to the left offset, that offset is naturally 1, and then do | with the
     * previous data, so that offset is replaced with 1.
     *
     * @param num the num
     */
    public void add(int num) {
      bits[getIndex(num)] |= getOffsetBit(num);
    }

    /**
     * Clear.
     *
     * <p>After moving 1 to the left offset, that offset is naturally 1, and then reverse it, and do
     * & with the current value to clear the current offset.
     *
     * @param num the num
     */
    public void clear(int num) {
      bits[getIndex(num)] &= ~getOffsetBit(num);
    }

    /**
     * Contain boolean.
     *
     * <p>After shifting 1 to the left offset, that offset is naturally 1, and then do & with the
     * previous data to determine whether it is 0
     *
     * @param num the num
     * @return the boolean
     */
    public boolean contain(int num) {
      return (bits[getIndex(num)] & getOffsetBit(num)) != 0;
    }

    // num/8 gets the index of byte[]
    private int getIndex(int num) {
      return num >> 3;
    }

    // num%8 gets the position of byte[index]
    private int getOffsetBit(int num) {
      return 1 << (num & 0x07);
    }
  }

  /**
   * Huffman Encoding，适用于已知压缩
   *
   * @author cenghui
   */
  public class Huffman {
    /**
     * Build huffman tree.
     *
     * @param text the text
     */
    // Utility function to check if Huffman Tree contains only a single node
    // Builds Huffman Tree and decodes the given input text
    public void buildHuffmanTree(String text) {
      // Base case: empty string
      if (text == null || text.length() == 0) return;
      // Count the frequency of appearance of each character and store it in a map
      Map<Character, Integer> freq = new HashMap<>();
      for (char c : text.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);
      // create a priority queue to store live nodes of the Huffman tree.
      // Notice that the highest priority item has the lowest frequency
      PriorityQueue<Node> pq;
      pq = new PriorityQueue<>(Comparator.comparingInt(l -> l.freq));
      // create a leaf node for each character and add it to the priority queue.
      for (Map.Entry<Character, Integer> entry : freq.entrySet())
        pq.add(new Node(entry.getKey(), entry.getValue()));
      // do till there is more than one node in the queue
      while (pq.size() != 1) {
        // Remove the two nodes of the highest priority (the lowest frequency) from the queue
        Node left = pq.poll();
        Node right = pq.poll();
        // create a new internal node with these two nodes as children and with a frequency equal to
        // the sum of both nodes' frequencies.
        // Add the new node to the priority queue.
        int sum = left.freq + right.freq;
        pq.add(new Node(null, sum, left, right));
      }
      // `root` stores pointer to the root of Huffman Tree
      Node root = pq.peek();
      // Traverse the Huffman tree and store the Huffman codes in a map
      Map<Character, String> huffmanCode = new HashMap<>();
      encode(root, "", huffmanCode);
      // Print the Huffman codes
      System.out.println("Huffman Codes are: " + huffmanCode);
      System.out.println("The original string is: " + text);
      // Print encoded string
      StringBuilder sb = new StringBuilder();
      for (char c : text.toCharArray()) sb.append(huffmanCode.get(c));
      System.out.println("The encoded string is: " + sb);
      System.out.print("The decoded string is: ");
      // Special case: For input like a, aa, aaa, etc.
      if (root.left == null && root.right == null)
        while (root.freq-- > 0) System.out.print(root.ch);
      else {
        // Traverse the Huffman Tree again and this time,
        // decode the encoded string
        int index = -1;
        while (index < sb.length() - 1) index = decode(root, index, sb);
      }
    }

    // Traverse the Huffman Tree and store Huffman Codes in a map.
    private void encode(Node root, String str, Map<Character, String> huffmanCode) {
      if (root == null) return;
      // Found a leaf node
      if (root.left == null && root.right == null)
        huffmanCode.put(root.ch, str.length() > 0 ? str : "1");
      encode(root.left, str + '0', huffmanCode);
      encode(root.right, str + '1', huffmanCode);
    }

    // Traverse the Huffman Tree and decode the encoded string
    private int decode(Node root, int index, StringBuilder sb) {
      if (root == null) return index;
      // Found a leaf node
      if (root.left == null && root.right == null) {
        System.out.print(root.ch);
        return index;
      }
      index += 1;
      root = (sb.charAt(index) == '0') ? root.left : root.right;
      index = decode(root, index, sb);
      return index;
    }

    private class Node {
      /** The Ch. */
      Character ch;

      /** The Freq. */
      int freq;

      /** The Left. */
      Node left = null,
          /** The Right. */
          right = null;

      /**
       * Instantiates a new Node.
       *
       * @param ch the ch
       * @param freq the freq
       */
      public Node(Character ch, int freq) {
        this.ch = ch;
        this.freq = freq;
      }

      /**
       * Instantiates a new Node.
       *
       * @param ch the ch
       * @param freq the freq
       * @param left the left
       * @param right the right
       */
      public Node(Character ch, int freq, Node left, Node right) {
        this.ch = ch;
        this.freq = freq;
        this.left = left;
        this.right = right;
      }
    }
  }

  /**
   * 短网址设计 https://www.geeksforgeeks.org/system-design-url-shortening-service/
   *
   * <p>Characteristics - 高可用 & 低延迟
   *
   * <p>Implementation
   *
   * <p>时延要求，且关系一一对应，因此 NoSQL 优先
   *
   * <p>高可用，需要分布式与定期归档
   *
   * @author cenghui
   */
  public class TinyUrl {
    /** The Long urls. */
    // tinyUrl:longUrl
    Map<String, UrlItem> longUrls = new ConcurrentHashMap<String, UrlItem>(10);

    /**
     * Generate string.
     *
     * @param longUrl the long url
     * @return the string
     */
    public String generate(String longUrl) {
      if (exist(longUrl)) return longUrls.get(longUrl).longUrl;
      String tinyUrl = encoding(longUrl);
      longUrls.put(tinyUrl, new UrlItem());
      return tinyUrl;
    }

    /**
     * Redirect string.
     *
     * @param tinyUrl the tiny url
     * @return the string
     */
    public String redirect(String tinyUrl) {
      return tinyUrl;
    }

    /**
     * Exist boolean.
     *
     * @param longUrl the long url
     * @return the boolean
     */
    public boolean exist(String longUrl) {
      return Objects.equals(longUrl, " ");
    }

    // 生成全局唯一 ID 有如下方式
    // 1.MD5，前提条件是存储组件支持 set if not exist
    // 2.全局自增计数器，显然单点存在两个 QPS & crash 两个瓶颈，因此需要引入分布式
    // 假如 NodeA 生成 1-1k，而 NodeB 生成 1k1-2k，其一宕机需要保证可用，因此通过 Redis 提供的 INCR & cluster 即可
    private String encoding(String longUrl) {
      return longUrl;
    }

    private String decoding(String tinyUrl) {
      return tinyUrl;
    }

    private class UrlItem {
      /** The Long url. */
      String longUrl;
    }
  }
}

/**
 * 三种常用的限流算法
 *
 * <p>窗口
 *
 * <p>漏桶
 *
 * <p>令牌桶
 */
class Limiter {
  /** The interface L limiter. */
  public interface LLimiterImpl {
    /**
     * Try acquire boolean.
     *
     * @param needTokenNum the need token num
     * @return the boolean
     */
    boolean tryAcquire(long needTokenNum);
  }

  /**
   * 滑动窗口限流算法
   *
   * <p>适用于流量平滑 & 抖动可预知
   *
   * @author cenghui
   */
  public class SlidingWindow implements LLimiterImpl {
    // 每分钟限制请求数
    private final long permitsPerMinute;
    // 计数器, k-为当前窗口的开始时间值秒，value为当前窗口的计数
    private final Map<Long, Long> counters = new TreeMap<>();

    /**
     * Instantiates a new Sliding window.
     *
     * @param permitsPerMinute the permits per minute
     */
    public SlidingWindow(long permitsPerMinute) {
      this.permitsPerMinute = permitsPerMinute;
    }

    @Override
    public synchronized boolean tryAcquire(long needTokenNum) {
      // 获取当前时间的所在的子窗口值； 10s一个窗口
      long currentWindowTime = LocalDateTime.now().toEpochSecond(ZoneOffset.UTC) / 10 * 10;
      // 获取当前窗口的请求总量
      int currentWindowCount = getCurrentWindowCount(currentWindowTime);
      if (currentWindowCount >= permitsPerMinute) return false;
      // 计数器 + 1
      counters.merge(currentWindowTime, needTokenNum, Long::sum);
      return true;
    }
    /**
     * 获取当前窗口中的所有请求数（并删除所有无效的子窗口计数器）
     *
     * @param currentWindowTime 当前子窗口时间
     * @return 当前窗口中的计数
     */
    private int getCurrentWindowCount(long currentWindowTime) {
      // 计算出窗口的开始位置时间
      long startTime = currentWindowTime - 50;
      int res = 0;
      // 遍历当前存储的计数器，删除无效的子窗口计数器，并累加当前窗口中的所有计数器之和
      Iterator<Map.Entry<Long, Long>> iterator = counters.entrySet().iterator();
      while (iterator.hasNext()) {
        Map.Entry<Long, Long> entry = iterator.next();
        if (entry.getKey() < startTime) iterator.remove();
        else res += entry.getValue();
      }
      return res;
    }
  }

  /**
   * 令牌桶算法
   *
   * <p>适用于流量突发
   *
   * @author cenghui
   */
  public class TokenBucket implements LLimiterImpl {
    private final double unitAddNum; // 单位时间往桶中放令牌数量
    private final long peak; // 峰值，即桶中令牌上限
    private final AtomicLong currentTokenCount = new AtomicLong(0); // 桶内剩余令牌
    private volatile long nextRefreshTime; // 下一次刷新桶中令牌数量的时间戳
    private volatile long lastAcquireTime; // 上一次从桶中获取令牌的时间戳
    /**
     * 初始化
     *
     * @param unitAddNum 1秒增加令牌量
     * @param maxToken 桶内令牌上限
     */
    public TokenBucket(double unitAddNum, long maxToken) {
      this.unitAddNum = unitAddNum;
      this.peak = maxToken;
      this.nextRefreshTime = calNextRefreshTime(System.currentTimeMillis());
    }

    @Override
    /**
     * 批量获取令牌
     *
     * <p>1.特判量级，通过后加锁修改 currentTokenCount
     *
     * <p>2.获取当前时间戳，与上次做差计算剩余令牌数，并更新当前上限
     *
     * <p>3.假如量级足够，则更新 currentTokenCount，否则失败
     *
     * @param needTokenNum the need token num
     * @return the boolean
     */
    public boolean tryAcquire(long required) {
      if (required > this.peak) return false;
      synchronized (this) {
        long currentTimestamp = System.currentTimeMillis();
        this.refreshCurrentTokenCount(currentTimestamp);
        return required <= this.currentTokenCount.get()
            && this.doAcquire(required, currentTimestamp);
      }
    }

    private boolean doAcquire(long required, long currentTimestamp) {
      final int maxTries = 3;
      long old;
      for (int i = 0; i < maxTries; i++) {
        old = this.currentTokenCount.get();
        if (this.currentTokenCount.compareAndSet(old, (old - required))) {
          this.lastAcquireTime = currentTimestamp;
          return true;
        }
      }
      return false;
    }
    // 刷新桶中令牌数量，惰性
    private void refreshCurrentTokenCount(long currentTimestamp) {
      if (this.nextRefreshTime > currentTimestamp) return;
      long addOneMs = Math.round(1.0D / this.unitAddNum * 1000D); // 这么久才能加1个令牌
      this.currentTokenCount.set(
          Math.min(
              this.peak,
              this.currentTokenCount.get()
                  + 1
                  + (currentTimestamp - this.nextRefreshTime) / addOneMs)); // 计算当前需要添加的令牌量
      this.nextRefreshTime = calNextRefreshTime(currentTimestamp); // 计算下一次添加
    }
    // 计算下一次添加
    private long calNextRefreshTime(long currentTimestamp) {
      if (currentTimestamp < this.nextRefreshTime) return this.nextRefreshTime;
      long addOneMs = Math.round(1.0D / this.unitAddNum * 1000D); // 这么久才能加1个令牌
      return currentTimestamp + addOneMs;
    }
  }
}

/** 容器类 */
class Container {}

/** 缓存类，LRU & LFU 参考 OOthers */
class CCache {
  private final int capacity;
  private final Map<String, CacheItem> data = new HashMap<String, CacheItem>();

  /**
   * Instantiates a new C cache.
   *
   * @param capacity the capacity
   */
  public CCache(int capacity) {
    this.capacity = capacity;
    new Thread(this::gcWorker);
  }

  /**
   * Get object.
   *
   * @param key the key
   * @return the object
   */
  public Object get(String key) {
    return data.get(key).value;
  }

  /**
   * Set.
   *
   * @param key the key
   * @param val the val
   */
  public void set(String key, Object val) {
    data.put(key, new CacheItem(key, val));
  }

  /**
   * Del.
   *
   * @param key the key
   */
  public void del(String key) {}

  /**
   * Expire.
   *
   * @param key the key
   * @param duration the duration
   */
  public void expire(String key, Long duration) {}

  private void gcWorker() {}

  private class CacheItem {
    private final Long createdAt;
    /** The Key. */
    String key;
    /** The Value. */
    Object value;
    /** The Expire. */
    long expire;

    private int lru;
    private boolean deleted;

    /**
     * Instantiates a new Cache item.
     *
     * @param key the key
     * @param value the value
     */
    public CacheItem(String key, Object value) {
      this.key = key;
      this.value = value;
      this.expire = Integer.MAX_VALUE;
      this.createdAt = System.currentTimeMillis();
    }

    /**
     * Instantiates a new Cache item.
     *
     * @param key the key
     * @param value the value
     * @param expire the expire
     */
    public CacheItem(String key, Object value, Long expire) {
      this.key = key;
      this.value = value;
      this.expire = expire;
      this.createdAt = System.currentTimeMillis();
    }

    /**
     * Is expired boolean.
     *
     * @param currentTimeStamp the current time stamp
     * @return the boolean
     */
    public boolean isExpired(Long currentTimeStamp) {
      return currentTimeStamp - currentTimeStamp > expire;
    }

    /**
     * Del boolean.
     *
     * @return the boolean
     */
    public boolean del() {
      if (deleted) return false;
      return deleted = true;
    }
  }
}

/** 参考 CMU 15445 https://zhenghe.gitbook.io/open-courses/cmu-15-445-645-database-systems */
class Databases {
  /** 存储层 */
  public class Storage {
    /** The type Index. */
    class Index {}

    /** The type Buffer pool. */
    class BufferPool {}

    /** The type Logger. */
    class Logger {}
  }

  /** 与 client 连接并交互 */
  public class Handler {
    /** The type Searcher. */
    class Searcher {}

    /** The type Dm ler. */
    class DMLer {}

    /** The type Locker. */
    class Locker {}
  }

  /** 维护可用性 */
  public class Replication {
    /** The type Dumper. */
    class Dumper {}

    /** The type Relay er. */
    class RelayEr {}
  }
}

/** Nginx 提供的三种负载均衡算法 */
class LoadBalancing {
  /** 轮询 */
  public class RoundRobin {}
  /** 哈希取模 */
  public class Hash {}
  /** 最小连接 */
  public class LeastConnection {}
}
