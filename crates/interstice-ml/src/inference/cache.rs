use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use parking_lot::RwLock;

/// Thread-safe LRU Cache implementation
pub struct LRUCache<K, V> 
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    capacity: usize,
    map: HashMap<K, (V, usize)>, // Value and access order
    order: VecDeque<K>,
    access_counter: usize,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
            access_counter: 0,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some((value, _)) = self.map.get_mut(key) {
            self.access_counter += 1;
            let value = value.clone();
            
            // Update access order
            self.order.retain(|k| k != key);
            self.order.push_back(key.clone());
            
            // Update access counter in map
            self.map.get_mut(key).unwrap().1 = self.access_counter;
            
            Some(value)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Evict least recently used
            if let Some(lru_key) = self.order.pop_front() {
                self.map.remove(&lru_key);
            }
        }
        
        self.access_counter += 1;
        self.map.insert(key.clone(), (value, self.access_counter));
        self.order.retain(|k| k != &key);
        self.order.push_back(key);
    }
    
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
        self.access_counter = 0;
    }
    
    pub fn len(&self) -> usize {
        self.map.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }
}

/// Thread-safe wrapper for LRU Cache
pub struct ConcurrentLRUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    inner: Arc<RwLock<LRUCache<K, V>>>,
}

impl<K, V> ConcurrentLRUCache<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LRUCache::new(capacity))),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.write().get(key)
    }

    pub fn put(&self, key: K, value: V) {
        self.inner.write().put(key, value);
    }
    
    pub fn clear(&self) {
        self.inner.write().clear();
    }
}