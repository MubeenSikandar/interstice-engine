pub mod traits;
pub mod slack;

pub use slack::SlackAdapter;

pub use traits::{PlatformAdapter, PlatformResponse};

use interstice_core::Platform;
use std::collections::HashMap;

/// Manages all platform adapters
pub struct AdapterManager {
    adapters: HashMap<Platform, Box<dyn PlatformAdapter>>,
}

impl AdapterManager {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }

    pub fn register(&mut self, adapter: Box<dyn PlatformAdapter>) {
        self.adapters.insert(adapter.platform(), adapter);
    }

    pub fn get(&self, platform: Platform) -> Option<&dyn PlatformAdapter> {
        self.adapters.get(&platform).map(|b| b.as_ref())
    }
}
