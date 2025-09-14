//interstice-ml/src/inference/cache.rs
//! High-Performance LRU Cache Implementation
//! 
//! This module provides production-ready LRU (Least Recently Used) cache implementations
//! with advanced features including TTL support, statistics, async interfaces, and more.


use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::{BuildHasher, Hash};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Custom error types for cache operations
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Cache capacity must be greater than 0")]
    InvalidCapacity,
    
    #[error("Key not found in cache")]
    KeyNotFound,
    
    #[error("Value expired")]
    ValueExpired,
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Cache entry metadata
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
}

impl<V> CacheEntry<V> {
    fn new(value: V, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            value,
            inserted_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        }
    }
    
    fn is_expired(&self) -> bool {
        self.ttl.map_or(false, |ttl| self.inserted_at.elapsed() > ttl)
    }
    
    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Node in the doubly-linked list for LRU tracking
struct Node<K> {
    key: K,
    prev: Option<usize>,
    next: Option<usize>,
}

impl<K> Node<K> {
    fn new(key: K) -> Self {
        Self {
            key,
            prev: None,
            next: None,
        }
    }
}

/// Doubly-linked list for efficient LRU tracking using indices
struct LinkedList<K> {
    nodes: Vec<Node<K>>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl<K> LinkedList<K> {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            head: None,
            tail: None,
        }
    }
    
    /// Push a node to the front (most recently used)
    fn push_front(&mut self, key: K) -> usize {
        let node = Node::new(key);
        let index = self.nodes.len();
        self.nodes.push(node);
        
        if let Some(head_idx) = self.head {
            self.nodes[head_idx].prev = Some(index);
        }
        
        self.nodes[index].next = self.head;
        self.head = Some(index);
        
        if self.tail.is_none() {
            self.tail = Some(index);
        }
        
        index
    }
    
    /// Remove a node from the list
    fn unlink(&mut self, node_idx: usize) {
        if let Some(node) = self.nodes.get(node_idx) {
            let prev = node.prev;
            let next = node.next;
            
            if let Some(prev_idx) = prev {
                if let Some(prev_node) = self.nodes.get_mut(prev_idx) {
                    prev_node.next = next;
                }
            } else {
                self.head = next;
            }
            
            if let Some(next_idx) = next {
                if let Some(next_node) = self.nodes.get_mut(next_idx) {
                    next_node.prev = prev;
                }
            } else {
                self.tail = prev;
            }
        }
    }
    
    /// Pop the back node (least recently used)
    fn pop_back(&mut self) -> Option<K> {
        self.tail.map(|tail_idx| {
            self.unlink(tail_idx);
            self.nodes.remove(tail_idx).key
        })
    }
}

impl<K> Drop for LinkedList<K> {
    fn drop(&mut self) {
        // Vec will automatically clean up the nodes
    }
}

/// Configuration for LRU cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache
    pub capacity: usize,
    
    /// Default TTL for entries (None = no expiration)
    pub default_ttl: Option<Duration>,
    
    /// Enable automatic cleanup of expired entries
    pub auto_cleanup: bool,
    
    /// Interval for automatic cleanup (if enabled)
    pub cleanup_interval: Duration,
    
    /// Track detailed statistics
    pub enable_statistics: bool,
    
    /// Initial capacity for internal HashMap
    pub initial_capacity: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            default_ttl: None,
            auto_cleanup: false,
            cleanup_interval: Duration::from_secs(60),
            enable_statistics: true,
            initial_capacity: None,
        }
    }
}

/// Cache statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub insertions: AtomicU64,
    pub evictions: AtomicU64,
    pub expirations: AtomicU64,
    pub current_size: AtomicUsize,
}

impl CacheStatistics {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            insertions: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
            current_size: AtomicUsize::new(0),
        }
    }
    
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let total = hits + self.misses.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
    
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.insertions.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.expirations.store(0, Ordering::Relaxed);
    }
}

impl fmt::Display for CacheStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheStats {{ hits: {}, misses: {}, hit_rate: {:.2}%, size: {}, evictions: {} }}",
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.hit_rate() * 100.0,
            self.current_size.load(Ordering::Relaxed),
            self.evictions.load(Ordering::Relaxed)
        )
    }
}

/// Optimized LRU Cache using a HashMap and doubly-linked list
pub struct LRUCache<K, V, S = std::collections::hash_map::RandomState>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    capacity: usize,
    map: HashMap<K, (CacheEntry<V>, usize), S>,
    list: LinkedList<K>,
    config: CacheConfig,
    stats: Option<Arc<CacheStatistics>>,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Creates a new LRU cache with the specified capacity
    pub fn new(capacity: usize) -> Result<Self, CacheError> {
        if capacity == 0 {
            return Err(CacheError::InvalidCapacity);
        }
        
        let config = CacheConfig {
            capacity,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Creates a new LRU cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Result<Self, CacheError> {
        if config.capacity == 0 {
            return Err(CacheError::InvalidCapacity);
        }
        
        let initial_capacity = config.initial_capacity.unwrap_or(config.capacity);
        let stats = if config.enable_statistics {
            Some(Arc::new(CacheStatistics::new()))
        } else {
            None
        };
        
        Ok(Self {
            capacity: config.capacity,
            map: HashMap::with_capacity(initial_capacity),
            list: LinkedList::new(),
            config,
            stats,
        })
    }
}

impl<K, V, S> LRUCache<K, V, S>
where
    K: Hash + Eq + Clone,
    V: Clone,
    S: BuildHasher,
{
    /// Gets a value from the cache, updating its position in the LRU order
    pub fn get<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        // Check if the key exists and if the entry is not expired
        let needs_removal = self.map.get(key).map_or(false, |(entry, _)| entry.is_expired());
        
        if needs_removal {
            self.remove_internal(key);
            if let Some(ref stats) = self.stats {
                stats.expirations.fetch_add(1, Ordering::Relaxed);
                stats.misses.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        }
        
        // First, get the key and value without holding a mutable reference
        let (actual_key, old_node_idx) = if let Some((k, (_, node_idx))) = self.map.get_key_value(key) {
            (k.clone(), *node_idx)
        } else {
            if let Some(ref stats) = self.stats {
                stats.misses.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        };
        
        // Now we can safely get a mutable reference and update
        if let Some((entry, node_idx)) = self.map.get_mut(key) {
            entry.touch();
            let value = entry.value.clone();
            
            // Move node to front (most recently used)
            self.list.unlink(old_node_idx);
            let new_idx = self.list.push_front(actual_key);
            *node_idx = new_idx;
            
            if let Some(ref stats) = self.stats {
                stats.hits.fetch_add(1, Ordering::Relaxed);
            }
            
            Some(value)
        } else {
            if let Some(ref stats) = self.stats {
                stats.misses.fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    }
    
    /// Gets a value without updating LRU order (peek operation)
    pub fn peek<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key).and_then(|(entry, _)| {
            if entry.is_expired() {
                None
            } else {
                Some(&entry.value)
            }
        })
    }
    
    /// Inserts a key-value pair into the cache
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        self.put_with_ttl(key, value, self.config.default_ttl)
    }
    
    /// Inserts a key-value pair with a specific TTL
    pub fn put_with_ttl(&mut self, key: K, value: V, ttl: Option<Duration>) -> Option<V> {
        // Check if key already exists
        if let Some((old_entry, old_node_idx)) = self.map.remove(&key) {
            self.list.unlink(old_node_idx);
            
            // Insert new entry
            let node_idx = self.list.push_front(key.clone());
            self.map.insert(key, (CacheEntry::new(value, ttl), node_idx));
            
            if let Some(ref stats) = self.stats {
                stats.insertions.fetch_add(1, Ordering::Relaxed);
            }
            
            return Some(old_entry.value);
        }
        
        // Check if we need to evict
        if self.map.len() >= self.capacity {
            self.evict_lru();
        }
        
        // Insert new entry
        let node_idx = self.list.push_front(key.clone());
        self.map.insert(key, (CacheEntry::new(value, ttl), node_idx));
        
        if let Some(ref stats) = self.stats {
            stats.insertions.fetch_add(1, Ordering::Relaxed);
            stats.current_size.store(self.map.len(), Ordering::Relaxed);
        }
        
        None
    }
    
    /// Removes a key from the cache
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_internal(key)
    }
    
    fn remove_internal<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some((entry, node_idx)) = self.map.remove(key) {
            self.list.unlink(node_idx);
            
            if let Some(ref stats) = self.stats {
                stats.current_size.store(self.map.len(), Ordering::Relaxed);
            }
            
            Some(entry.value)
        } else {
            None
        }
    }
    
    /// Evicts the least recently used entry
    fn evict_lru(&mut self) {
        if let Some(key) = self.list.pop_back() {
            // Adjust indices in map after removal
            let removed_idx = self.list.nodes.len(); // Index of removed node
            for (_, node_idx) in self.map.values_mut() {
                if *node_idx > removed_idx {
                    *node_idx -= 1;
                }
            }
            
            self.map.remove(&key);
            
            if let Some(ref stats) = self.stats {
                stats.evictions.fetch_add(1, Ordering::Relaxed);
                stats.current_size.store(self.map.len(), Ordering::Relaxed);
            }
        }
    }
    
    /// Removes all expired entries
    pub fn cleanup_expired(&mut self) -> usize {
        let expired_keys: Vec<K> = self.map
            .iter()
            .filter(|(_, (entry, _))| entry.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        
        let count = expired_keys.len();
        
        for key in expired_keys {
            self.remove_internal(&key);
            if let Some(ref stats) = self.stats {
                stats.expirations.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        count
    }
    
    /// Clears all entries from the cache
    pub fn clear(&mut self) {
        self.map.clear();
        self.list = LinkedList::new();
        
        if let Some(ref stats) = self.stats {
            stats.current_size.store(0, Ordering::Relaxed);
        }
    }
    
    /// Returns the current number of entries in the cache
    pub fn len(&self) -> usize {
        self.map.len()
    }
    
    /// Checks if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    
    /// Checks if a key exists in the cache (without updating LRU)
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key).map_or(false, |(entry, _)| !entry.is_expired())
    }
    
    /// Returns the cache capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Returns the cache configuration
    pub fn config(&self) -> &CacheConfig {
        &self.config
    }
    
    /// Resizes the cache to a new capacity
    pub fn resize(&mut self, new_capacity: usize) -> Result<(), CacheError> {
        if new_capacity == 0 {
            return Err(CacheError::InvalidCapacity);
        }
        
        self.capacity = new_capacity;
        self.config.capacity = new_capacity;
        
        // Evict entries if necessary
        while self.map.len() > self.capacity {
            self.evict_lru();
        }
        
        Ok(())
    }
    
    /// Returns cache statistics (if enabled)
    pub fn statistics(&self) -> Option<Arc<CacheStatistics>> {
        self.stats.clone()
    }
    
    /// Returns an iterator over the cache entries
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.map.iter()
            .filter(|(_, (entry, _))| !entry.is_expired())
            .map(|(k, (entry, _))| (k, &entry.value))
    }
}

/// Thread-safe concurrent LRU cache
pub struct ConcurrentLRUCache<K, V, S = std::collections::hash_map::RandomState>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
    S: BuildHasher + Send + Sync,
{
    inner: Arc<RwLock<LRUCache<K, V, S>>>,
    stats: Option<Arc<CacheStatistics>>,
    cleanup_handle: Option<Arc<AtomicU64>>, // For cleanup thread coordination
}

impl<K, V> ConcurrentLRUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Creates a new concurrent LRU cache
    pub fn new(capacity: usize) -> Result<Self, CacheError> {
        let config = CacheConfig {
            capacity,
            ..Default::default()
        };
        
        Self::with_config(config)
    }
    
    /// Creates a new concurrent LRU cache with custom configuration
    pub fn with_config(config: CacheConfig) -> Result<Self, CacheError> {
        let stats = if config.enable_statistics {
            Some(Arc::new(CacheStatistics::new()))
        } else {
            None
        };
        
        let cache = LRUCache::with_config(config.clone())?;
        let inner = Arc::new(RwLock::new(cache));
        
        let cleanup_handle = if config.auto_cleanup {
            let handle = Arc::new(AtomicU64::new(0));
            let handle_clone = handle.clone();
            let inner_clone = inner.clone();
            let interval = config.cleanup_interval;
            
            // Spawn cleanup thread
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(interval);
                    
                    // Check if we should stop
                    if handle_clone.load(Ordering::Relaxed) == u64::MAX {
                        break;
                    }
                    
                    inner_clone.write().cleanup_expired();
                }
            });
            
            Some(handle)
        } else {
            None
        };
        
        Ok(Self {
            inner,
            stats,
            cleanup_handle,
        })
    }
    
    /// Gets a value from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.write().get(key)
    }
    
    /// Gets a value without updating LRU order
    pub fn peek(&self, key: &K) -> Option<V> {
        self.inner.read().peek(key).cloned()
    }
    
    /// Inserts a key-value pair
    pub fn put(&self, key: K, value: V) -> Option<V> {
        self.inner.write().put(key, value)
    }
    
    /// Inserts a key-value pair with TTL
    pub fn put_with_ttl(&self, key: K, value: V, ttl: Option<Duration>) -> Option<V> {
        self.inner.write().put_with_ttl(key, value, ttl)
    }
    
    /// Removes a key from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.write().remove(key)
    }
    
    /// Clears all entries
    pub fn clear(&self) {
        self.inner.write().clear();
    }
    
    /// Returns the current size
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }
    
    /// Checks if empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
    
    /// Checks if a key exists
    pub fn contains_key(&self, key: &K) -> bool {
        self.inner.read().contains_key(key)
    }
    
    /// Manually triggers cleanup of expired entries
    pub fn cleanup_expired(&self) -> usize {
        self.inner.write().cleanup_expired()
    }
    
    /// Returns cache statistics
    pub fn statistics(&self) -> Option<Arc<CacheStatistics>> {
        self.stats.clone().or_else(|| self.inner.read().statistics())
    }
    
    /// Batch get operation
    pub fn get_multiple(&self, keys: &[K]) -> Vec<Option<V>> {
        let mut cache = self.inner.write();
        keys.iter().map(|k| cache.get(k)).collect()
    }
    
    /// Batch put operation
    pub fn put_multiple(&self, entries: Vec<(K, V)>) -> Vec<Option<V>> {
        let mut cache = self.inner.write();
        entries.into_iter()
            .map(|(k, v)| cache.put(k, v))
            .collect()
    }
    
    /// Executes a function with read access to the cache
    pub fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&LRUCache<K, V>) -> R,
    {
        let cache = self.inner.read();
        f(&*cache)
    }
    
    /// Executes a function with write access to the cache
    pub fn with_write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut LRUCache<K, V>) -> R,
    {
        let mut cache = self.inner.write();
        f(&mut *cache)
    }
}

impl<K, V, S> Drop for ConcurrentLRUCache<K, V, S>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
    S: BuildHasher + Send + Sync,
{
    fn drop(&mut self) {
        // Signal cleanup thread to stop
        if let Some(ref handle) = self.cleanup_handle {
            handle.store(u64::MAX, Ordering::Relaxed);
        }
    }
}

impl<K, V, S> Clone for ConcurrentLRUCache<K, V, S>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
    S: BuildHasher + Send + Sync,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            stats: self.stats.clone(),
            cleanup_handle: self.cleanup_handle.clone(),
        }
    }
}

/// Builder pattern for cache configuration
pub struct CacheBuilder {
    config: CacheConfig,
}

impl CacheBuilder {
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
        }
    }
    
    pub fn capacity(mut self, capacity: usize) -> Self {
        self.config.capacity = capacity;
        self
    }
    
    pub fn default_ttl(mut self, ttl: Duration) -> Self {
        self.config.default_ttl = Some(ttl);
        self
    }
    
    pub fn auto_cleanup(mut self, interval: Duration) -> Self {
        self.config.auto_cleanup = true;
        self.config.cleanup_interval = interval;
        self
    }
    
    pub fn with_statistics(mut self) -> Self {
        self.config.enable_statistics = true;
        self
    }
    
    pub fn initial_capacity(mut self, capacity: usize) -> Self {
        self.config.initial_capacity = Some(capacity);
        self
    }
    
    pub fn build<K, V>(self) -> Result<ConcurrentLRUCache<K, V>, CacheError>
    where
        K: Hash + Eq + Clone + Send + Sync + 'static,
        V: Clone + Send + Sync + 'static,
    {
        ConcurrentLRUCache::with_config(self.config)
    }
    
    pub fn build_non_concurrent<K, V>(self) -> Result<LRUCache<K, V>, CacheError>
    where
        K: Hash + Eq + Clone,
        V: Clone,
    {
        LRUCache::with_config(self.config)
    }
}

impl Default for CacheBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_operations() {
        let mut cache = LRUCache::new(3).unwrap();
        
        // Test put and get
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), Some(2));
        assert_eq!(cache.get(&"c"), Some(3));
        assert_eq!(cache.get(&"d"), None);
    }
    
    #[test]
    fn test_lru_eviction() {
        let mut cache = LRUCache::new(3).unwrap();
        
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        
        // Access "a" to make it recently used
        cache.get(&"a");
        
        // Add "d", should evict "b"
        cache.put("d", 4);
        
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), None); // Evicted
        assert_eq!(cache.get(&"c"), Some(3));
        assert_eq!(cache.get(&"d"), Some(4));
    }
    
    #[test]
    fn test_ttl() {
        let mut cache = LRUCache::new(3).unwrap();
        
        // Insert with 100ms TTL
        cache.put_with_ttl("a", 1, Some(Duration::from_millis(100)));
        
        assert_eq!(cache.get(&"a"), Some(1));
        
        // Wait for expiration
        thread::sleep(Duration::from_millis(150));
        
        assert_eq!(cache.get(&"a"), None);
    }
    
    #[test]
    fn test_concurrent_access() {
        let cache = ConcurrentLRUCache::new(100).unwrap();
        let cache_clone = cache.clone();
        
        let handle = thread::spawn(move || {
            for i in 0..50 {
                cache_clone.put(i, i * 2);
            }
        });
        
        for i in 50..100 {
            cache.put(i, i * 2);
        }
        
        handle.join().unwrap();
        
        assert_eq!(cache.len(), 100);
        assert_eq!(cache.get(&25), Some(50));
        assert_eq!(cache.get(&75), Some(150));
    }
    
    #[test]
    fn test_statistics() {
        let config = CacheConfig {
            capacity: 3,
            enable_statistics: true,
            ..Default::default()
        };
        
        let mut cache = LRUCache::with_config(config).unwrap();
        let stats = cache.statistics().unwrap();
        
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        
        // Hits
        cache.get(&"a");
        cache.get(&"b");
        
        // Misses
        cache.get(&"d");
        cache.get(&"e");
        
        assert_eq!(stats.hits.load(Ordering::Relaxed), 2);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 2);
        assert_eq!(stats.insertions.load(Ordering::Relaxed), 3);
        assert_eq!(stats.hit_rate(), 0.5);
    }
    
    #[test]
    fn test_builder_pattern() {
        let cache = CacheBuilder::new()
            .capacity(100)
            .default_ttl(Duration::from_secs(60))
            .with_statistics()
            .build::<String, i32>()
            .unwrap();
        
        cache.put("test".to_string(), 42);
        assert_eq!(cache.get(&"test".to_string()), Some(42));
    }
    
    #[test]
    fn test_resize() {
        let mut cache = LRUCache::new(5).unwrap();
        
        for i in 0..5 {
            cache.put(i, i);
        }
        
        assert_eq!(cache.len(), 5);
        
        // Resize to smaller capacity
        cache.resize(3).unwrap();
        assert_eq!(cache.len(), 3);
        
        // Most recently used items should remain
        assert!(cache.contains_key(&2));
        assert!(cache.contains_key(&3));
        assert!(cache.contains_key(&4));
    }
    
    #[test]
    fn test_peek_operation() {
        let mut cache = LRUCache::new(3).unwrap();
        
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        
        // Peek doesn't affect LRU order
        assert_eq!(cache.peek(&"a"), Some(&1));
        
        // Add new item, "a" should be evicted (still LRU)
        cache.put("d", 4);
        assert_eq!(cache.get(&"a"), None);
    }
    
    #[test]
    fn test_batch_operations() {
        let cache = ConcurrentLRUCache::new(10).unwrap();
        
        let entries = vec![
            ("a", 1),
            ("b", 2),
            ("c", 3),
        ];
        
        cache.put_multiple(entries);
        
        let keys = vec!["a", "b", "c", "d"];
        let values = cache.get_multiple(&keys);
        
        assert_eq!(values[0], Some(1));
        assert_eq!(values[1], Some(2));
        assert_eq!(values[2], Some(3));
        assert_eq!(values[3], None);
    }
    
    #[test]
    fn test_node_linked_list_operations() {
        let mut list = LinkedList::<String>::new();
        
        // Test empty list
        assert!(list.pop_back().is_none());
        
        // Add nodes
        let idx1 = list.push_front("first".to_string());
        assert!(list.head.is_some());
        assert!(list.tail.is_some());
        assert_eq!(idx1, 0);
        
        // Pop back
        let popped = list.pop_back();
        assert_eq!(popped, Some("first".to_string()));
        assert!(list.head.is_none());
        assert!(list.tail.is_none());
    }
    
    #[test]
    fn test_auto_cleanup() {
        let config = CacheConfig {
            capacity: 10,
            default_ttl: Some(Duration::from_millis(100)),
            auto_cleanup: true,
            cleanup_interval: Duration::from_millis(150),
            enable_statistics: true,
            initial_capacity: Some(10),
        };
        
        let cache = ConcurrentLRUCache::with_config(config).unwrap();
        
        // Add items that will expire
        for i in 0..5 {
            cache.put(i, i * 2);
        }
        
        assert_eq!(cache.len(), 5);
        
        // Wait for auto cleanup
        thread::sleep(Duration::from_millis(300));
        
        // Items should be expired and cleaned up
        assert_eq!(cache.len(), 0);
    }
    
    #[test]
    fn test_config_access() {
        let config = CacheConfig {
            capacity: 50,
            default_ttl: Some(Duration::from_secs(30)),
            auto_cleanup: false,
            cleanup_interval: Duration::from_secs(60),
            enable_statistics: true,
            initial_capacity: Some(25),
        };
        
        let cache = LRUCache::<String, i32>::with_config(config.clone()).unwrap();
        
        assert_eq!(cache.config().capacity, 50);
        assert_eq!(cache.config().default_ttl, Some(Duration::from_secs(30)));
        assert_eq!(cache.config().auto_cleanup, false);
        assert_eq!(cache.config().enable_statistics, true);
        assert_eq!(cache.config().initial_capacity, Some(25));
    }
    
    #[test]
    fn test_builder_non_concurrent() {
        let cache = CacheBuilder::new()
            .capacity(50)
            .default_ttl(Duration::from_secs(10))
            .initial_capacity(25)
            .with_statistics()
            .build_non_concurrent::<String, String>()
            .unwrap();
        
        assert_eq!(cache.capacity(), 50);
        assert!(cache.statistics().is_some());
    }
    
    #[test]
    fn test_builder_with_auto_cleanup() {
        let cache = CacheBuilder::new()
            .capacity(10)
            .default_ttl(Duration::from_millis(100))
            .auto_cleanup(Duration::from_millis(150))
            .with_statistics()
            .build::<i32, i32>()
            .unwrap();
        
        // Add some items
        for i in 0..5 {
            cache.put(i, i * 2);
        }
        
        assert_eq!(cache.len(), 5);
        
        // Wait for auto cleanup
        thread::sleep(Duration::from_millis(300));
        
        // Items should be expired and cleaned up
        assert_eq!(cache.len(), 0);
    }
    
    #[test]
    fn test_builder_all_methods() {
        let cache = CacheBuilder::new()
            .capacity(100)
            .default_ttl(Duration::from_secs(30))
            .auto_cleanup(Duration::from_secs(60))
            .with_statistics()
            .initial_capacity(50)
            .build::<String, i32>()
            .unwrap();
        
        // Test that all configuration is applied
        cache.put("test".to_string(), 42);
        assert_eq!(cache.get(&"test".to_string()), Some(42));
        assert!(cache.statistics().is_some());
    }
}