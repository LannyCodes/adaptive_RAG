"""
多级缓存管理器
L1: 内存缓存（LRU + TTL，毫秒级响应）
L2: 磁盘持久化缓存（使用 diskcache，重启后保留）
L3: 语义缓存（基于 embedding 相似度匹配）
"""

import os
import time
import hashlib
import threading
from collections import OrderedDict
from typing import Optional, Any, Callable
from diskcache import Cache as DiskCache
import numpy as np


class MemoryCache:
    """内存 LRU 缓存（线程安全）"""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()  # key -> (value, expiry)
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            value, expiry = self._cache[key]
            if time.time() > expiry:
                del self._cache[key]
                return None
            # LRU: 移到末尾
            self._cache.move_to_end(key)
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._lock:
            # LRU: 如果已存在，先删除
            if key in self._cache:
                del self._cache[key]
            # 淘汰最久未使用的
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
            self._cache[key] = (value, expiry)
    
    def delete(self, key: str):
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


class SemanticCache:
    """语义缓存：存储 query embedding 和答案，检索时按相似度匹配"""
    
    def __init__(self, similarity_threshold: float = 0.92, cache_dir: str = "./data/cache/semantic"):
        self._threshold = similarity_threshold
        os.makedirs(cache_dir, exist_ok=True)
        self._store = DiskCache(os.path.join(cache_dir, "semantic_store"))
        self._lock = threading.RLock()
    
    def _make_key(self, question: str) -> str:
        return hashlib.md5(question.encode()).hexdigest()
    
    def get(self, question: str, query_embedding_fn: Optional[Callable] = None) -> Optional[str]:
        """查找语义相似的缓存结果"""
        # 1. 精确匹配
        key = self._make_key(question)
        exact = self._store.get(key)
        if exact is not None:
            return exact.get("answer")
        
        # 2. 语义近似匹配（需提供 embedding 函数）
        if query_embedding_fn is not None and self._store.get("embeddings"):
            try:
                query_vec = np.array(query_embedding_fn(question))
                if query_vec.ndim == 2:
                    query_vec = query_vec[0]
                    
                for stored_key in self._store.iterkeys():
                    if stored_key == "embeddings":
                        continue
                    entry = self._store.get(stored_key)
                    if not entry or "embedding" not in entry:
                        continue
                    stored_vec = np.array(entry["embedding"])
                    # 余弦相似度
                    norm = np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                    if norm == 0:
                        continue
                    similarity = float(np.dot(query_vec, stored_vec) / norm)
                    if similarity >= self._threshold:
                        return entry.get("answer")
            except Exception:
                pass  # 语义匹配失败，降级返回 None
        
        return None
    
    def set(self, question: str, answer: str, embedding: Optional[list] = None, ttl: int = 7200):
        """存入缓存"""
        key = self._make_key(question)
        entry = {"question": question, "answer": answer, "time": time.time()}
        if embedding is not None:
            entry["embedding"] = embedding
        self._store.set(key, entry, expire=ttl)
    
    def clear(self):
        self._store.clear()


class CacheManager:
    """
    缓存管理器（统一入口）
    
    用法:
        cache = CacheManager()
        
        # 写入
        cache.set_answer("什么是RAG？", "RAG是检索增强生成...")
        
        # 读取
        answer = cache.get_answer("什么是RAG？")
        
        # 语义匹配读取
        answer = cache.get_answer("RAG是什么？", query_embedding_fn=model.encode)
        
        # 写入检索结果缓存
        cache.set_retrieval("什么是RAG？", ["doc1", "doc2"])
    """
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # L1: 内存缓存（快速，有限容量）
        self._memory = MemoryCache(max_size=500, default_ttl=1800)
        
        # L2: 磁盘缓存（持久化）
        self._disk = DiskCache(os.path.join(cache_dir, "disk_cache"))
        
        # L3: 语义缓存
        self._semantic = SemanticCache(
            similarity_threshold=0.92,
            cache_dir=cache_dir
        )
        
        # L3: 检索结果缓存（磁盘持久化）
        self._retrieval_disk = DiskCache(os.path.join(cache_dir, "retrieval_cache"))
        
        print(f"  💾 缓存层已初始化 (内存 {self._memory._max_size}条 | 磁盘 ∞ | 语义阈值 0.92)")
    
    # ── 答案缓存 ──
    
    def get_answer(self, question: str,
                   query_embedding_fn: Optional[Callable] = None) -> Optional[str]:
        """获取缓存的答案"""
        key = hashlib.md5(question.encode()).hexdigest()
        
        # L1: 内存
        result = self._memory.get(f"ans:{key}")
        if result is not None:
            return result
        
        # L2: 磁盘
        result = self._disk.get(f"ans:{key}")
        if result is not None:
            self._memory.set(f"ans:{key}", result, ttl=600)
            return result
        
        # L3: 语义匹配
        result = self._semantic.get(question, query_embedding_fn)
        if result is not None:
            # 提升到 L1/L2
            self._memory.set(f"ans:{key}", result, ttl=600)
            self._disk.set(f"ans:{key}", result, expire=3600)
            return result
        
        return None
    
    def set_answer(self, question: str, answer: str,
                   embedding: Optional[list] = None, ttl: int = 3600):
        """缓存答案"""
        key = hashlib.md5(question.encode()).hexdigest()
        self._memory.set(f"ans:{key}", answer, ttl=ttl)
        self._disk.set(f"ans:{key}", answer, expire=ttl)
        if embedding is not None:
            self._semantic.set(question, answer, embedding=embedding, ttl=ttl)
    
    # ── 检索结果缓存 ──
    
    def get_retrieval(self, question: str) -> Optional[list]:
        """获取缓存的检索结果"""
        key = hashlib.md5(question.encode()).hexdigest()
        # L1 优先
        result = self._memory.get(f"ret:{key}")
        if result is not None:
            return result
        # L2
        result = self._retrieval_disk.get(f"ret:{key}")
        if result is not None:
            self._memory.set(f"ret:{key}", result, ttl=300)
        return result
    
    def set_retrieval(self, question: str, docs: list, ttl: int = 1800):
        """缓存检索结果（较短 TTL，因为知识库可能更新）"""
        key = hashlib.md5(question.encode()).hexdigest()
        self._memory.set(f"ret:{key}", docs, ttl=ttl)
        self._retrieval_disk.set(f"ret:{key}", docs, expire=ttl)
    
    # ── 缓存管理 ──
    
    def invalidate_question(self, question: str):
        """使指定问题的所有缓存失效"""
        key = hashlib.md5(question.encode()).hexdigest()
        self._memory.delete(f"ans:{key}")
        self._memory.delete(f"ret:{key}")
        self._disk.delete(f"ans:{key}")
        self._retrieval_disk.delete(f"ret:{key}")
    
    def clear_all(self):
        """清空所有缓存"""
        self._memory.clear()
        self._disk.clear()
        self._semantic.clear()
        self._retrieval_disk.clear()
        print("  🧹 所有缓存已清空")
    
    def stats(self) -> dict:
        """缓存统计"""
        return {
            "memory_cache_size": self._memory.size,
            "disk_cache_size": len(self._disk),
            "semantic_cache_size": len(self._semantic._store),
            "retrieval_cache_size": len(self._retrieval_disk),
        }
