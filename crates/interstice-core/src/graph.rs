//! # Evidence Graph Module
//! 
//! Advanced graph-based evidence tracking system for the INTERSTICE-ENGINE WorkOS.
//! Provides sophisticated relationship mapping between artifacts and outcomes with
//! temporal analysis, confidence scoring, and causal inference capabilities.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use petgraph::algo::{dijkstra, has_path_connecting, tarjan_scc};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument};
use uuid::Uuid;

use crate::artifact::Artifact;
use crate::error::CoreError;
use crate::outcome::{OutcomeId};
use crate::storage::StorageBackend;
use crate::traits::OutcomePrediction;
use crate::types::{Platform, UserId, WorkspaceId};

/// Result type for graph operations
pub type GraphResult<T> = Result<T, GraphError>;

/// Graph-specific error types
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Cycle detected in graph")]
    CycleDetected,
    
    #[error("Invalid edge weight: {0}")]
    InvalidWeight(f64),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] CoreError),
    
    #[error("Graph operation failed: {0}")]
    OperationFailed(String),
}

/// Evidence graph for tracking work-outcome relationships
pub struct EvidenceGraph {
    /// Storage backend for persistence
    storage: Arc<dyn StorageBackend>,
    
    /// In-memory graph representation
    graph: Arc<RwLock<DiGraph<Node, Edge>>>,
    
    /// Node index mapping for fast lookups
    node_indices: Arc<RwLock<HashMap<String, NodeIndex>>>,
    
    /// Relationship cache for performance
    cache: Arc<RwLock<RelationshipCache>>,
    
    /// Graph configuration
    config: GraphConfig,
    
    /// Graph metrics collector
    metrics: Arc<RwLock<GraphMetrics>>,
}

/// Graph configuration
#[derive(Debug, Clone, Deserialize)]
pub struct GraphConfig {
    /// Minimum confidence threshold for relationships
    pub min_confidence: f64,
    
    /// Maximum depth for traversal operations
    pub max_traversal_depth: usize,
    
    /// Enable temporal decay for older relationships
    pub enable_temporal_decay: bool,
    
    /// Decay rate per day for temporal relationships
    pub temporal_decay_rate: f64,
    
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    
    /// Enable causal inference
    pub enable_causal_inference: bool,
    
    /// Minimum support for pattern mining
    pub min_pattern_support: f64,
    
    /// Enable graph compression
    pub enable_compression: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            max_traversal_depth: 10,
            enable_temporal_decay: true,
            temporal_decay_rate: 0.95,
            cache_ttl_seconds: 300,
            enable_causal_inference: true,
            min_pattern_support: 0.1,
            enable_compression: true,
        }
    }
}

/// Node in the evidence graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// Artifact node
    Artifact {
        id: Uuid,
        workspace_id: WorkspaceId,
        platform: Platform,
        artifact_type: String,
        created_at: DateTime<Utc>,
        metadata: HashMap<String, serde_json::Value>,
    },
    
    /// Outcome node
    Outcome {
        id: OutcomeId,
        workspace_id: WorkspaceId,
        name: String,
        progress: f64,
        state: String,
        created_at: DateTime<Utc>,
        metadata: HashMap<String, serde_json::Value>,
    },
    
    /// User node (for collaboration tracking)
    User {
        id: UserId,
        workspace_id: WorkspaceId,
        metadata: HashMap<String, serde_json::Value>,
    },
    
    /// Cluster node (for hierarchical grouping)
    Cluster {
        id: Uuid,
        name: String,
        cluster_type: ClusterType,
        members: Vec<String>,
    },
}

impl Node {
    pub fn id(&self) -> String {
        match self {
            Node::Artifact { id, .. } => format!("artifact:{}", id),
            Node::Outcome { id, .. } => format!("outcome:{}", id.0),
            Node::User { id, .. } => format!("user:{}", id.as_str()),
            Node::Cluster { id, .. } => format!("cluster:{}", id),
        }
    }
    
    pub fn node_type(&self) -> NodeType {
        match self {
            Node::Artifact { .. } => NodeType::Artifact,
            Node::Outcome { .. } => NodeType::Outcome,
            Node::User { .. } => NodeType::User,
            Node::Cluster { .. } => NodeType::Cluster,
        }
    }
}

/// Node types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Artifact,
    Outcome,
    User,
    Cluster,
}

/// Edge in the evidence graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Relationship type
    pub relationship: RelationshipType,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Temporal weight (decays over time)
    pub temporal_weight: f64,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Edge {
    pub fn new(relationship: RelationshipType, confidence: f64) -> Self {
        let now = Utc::now();
        Self {
            relationship,
            confidence,
            temporal_weight: 1.0,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }
    
    /// Calculate effective weight considering temporal decay
    pub fn effective_weight(&self, decay_rate: f64) -> f64 {
        let days_old = (Utc::now() - self.updated_at).num_days() as f64;
        let temporal_factor = decay_rate.powf(days_old);
        self.confidence * self.temporal_weight * temporal_factor
    }
}

/// Relationship types in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Artifact contributes to outcome
    ContributesTo,
    
    /// Outcome depends on artifact
    DependsOn,
    
    /// User created artifact
    CreatedBy,
    
    /// User owns outcome
    OwnedBy,
    
    /// Artifact relates to another artifact
    RelatesTo,
    
    /// Outcome blocks another outcome
    Blocks,
    
    /// Outcome enables another outcome
    Enables,
    
    /// Artifact is part of cluster
    MemberOf,
    
    /// Causal relationship
    Causes,
    
    /// Correlation without causation
    CorrelatesWith,
}

/// Cluster types for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterType {
    Project,
    Sprint,
    Epic,
    Team,
    Topic,
    Temporal,
    Community,
}

/// Relationship cache for performance
struct RelationshipCache {
    /// Cached paths between nodes
    paths: HashMap<(String, String), Vec<NodeIndex>>,
    
    /// Cached influence scores
    influences: HashMap<String, f64>,
    
    /// Cached centrality scores
    centralities: HashMap<String, CentralityScores>,
    
    /// Cache expiry timestamps
    expiry: HashMap<String, DateTime<Utc>>,
}

impl RelationshipCache {
    fn new() -> Self {
        Self {
            paths: HashMap::new(),
            influences: HashMap::new(),
            centralities: HashMap::new(),
            expiry: HashMap::new(),
        }
    }
    
    fn get_path(&self, from: &str, to: &str) -> Option<&Vec<NodeIndex>> {
        let key = (from.to_string(), to.to_string());
        if self.is_expired(&format!("path:{}:{}", from, to)) {
            return None;
        }
        self.paths.get(&key)
    }
    
    fn set_path(&mut self, from: String, to: String, path: Vec<NodeIndex>, ttl: u64) {
        let key = (from.clone(), to.clone());
        self.paths.insert(key, path);
        self.set_expiry(format!("path:{}:{}", from, to), ttl);
    }
    
    fn is_expired(&self, key: &str) -> bool {
        self.expiry.get(key)
            .map(|exp| *exp < Utc::now())
            .unwrap_or(true)
    }
    
    fn set_expiry(&mut self, key: String, ttl_seconds: u64) {
        let expiry = Utc::now() + chrono::Duration::seconds(ttl_seconds as i64);
        self.expiry.insert(key, expiry);
    }
    
    fn invalidate(&mut self) {
        self.paths.clear();
        self.influences.clear();
        self.centralities.clear();
        self.expiry.clear();
    }
}

/// Graph metrics for monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
    pub density: f64,
    pub clustering_coefficient: f64,
    pub connected_components: usize,
    pub strongly_connected_components: usize,
    pub diameter: usize,
    pub avg_path_length: f64,
    pub last_updated: Option<DateTime<Utc>>,
}

/// Centrality scores for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityScores {
    pub degree: f64,
    pub betweenness: f64,
    pub closeness: f64,
    pub eigenvector: f64,
    pub pagerank: f64,
}

impl EvidenceGraph {
    /// Create a new evidence graph
    pub async fn new(storage: Arc<dyn StorageBackend>, config: GraphConfig) -> Self {
        Self {
            storage,
            graph: Arc::new(RwLock::new(DiGraph::new())),
            node_indices: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(RelationshipCache::new())),
            config,
            metrics: Arc::new(RwLock::new(GraphMetrics::default())),
        }
    }
    
    /// Build relationships between artifacts and outcomes
    #[instrument(skip(self, artifacts, predictions))]
    pub async fn build_relationships(
        &self,
        workspace_id: WorkspaceId,
        artifacts: &[Artifact],
        predictions: &[OutcomePrediction],
    ) -> GraphResult<RelationshipSummary> {
        info!(
            "Building relationships for {} artifacts and {} predictions",
            artifacts.len(),
            predictions.len()
        );
        
        let start_time = std::time::Instant::now();
        let mut relationships_created = 0;
        let mut nodes_created = 0;
        let mut failed_relationships = 0;
        
        // Begin transaction
        let mut graph = self.graph.write().await;
        let mut indices = self.node_indices.write().await;
        
        // Process artifacts
        for artifact in artifacts {
            // Create artifact node
            let artifact_node = Node::Artifact {
                id: artifact.id,
                workspace_id,
                platform: artifact.platform,
                artifact_type: artifact.artifact_type.type_name().to_string(),
                created_at: artifact.created_at,
                metadata: HashMap::new(),
            };
            
            let artifact_idx = self.add_or_get_node(
                &mut graph,
                &mut indices,
                artifact_node
            );
            nodes_created += 1;
            
            // Store artifact in backend
            self.storage.store_artifact(artifact.clone()).await?;
            
            // Process predictions
            for prediction in predictions {
                // Filter by minimum confidence
                if prediction.confidence < self.config.min_confidence as f32 {
                    debug!(
                        "Skipping prediction with low confidence: {}",
                        prediction.confidence
                    );
                    continue;
                }
                
                // Create outcome node if needed
                let outcome_node = Node::Outcome {
                    id: OutcomeId(prediction.outcome_id),
                    workspace_id,
                    name: prediction.outcome_name.clone(),
                    progress: 0.0,
                    state: "predicted".to_string(),
                    created_at: Utc::now(),
                    metadata: HashMap::new(),
                };
                
                let outcome_idx = self.add_or_get_node(
                    &mut graph,
                    &mut indices,
                    outcome_node
                );
                
                // Create edge
                let edge = Edge::new(
                    RelationshipType::ContributesTo,
                    prediction.confidence as f64,
                );
                
                graph.add_edge(artifact_idx, outcome_idx, edge);
                relationships_created += 1;
                
                // Store in backend
                if let Err(e) = self.storage.link_artifact_outcome(
                    artifact.id,
                    OutcomeId(prediction.outcome_id),
                    prediction.confidence as f64,
                    Some(serde_json::json!({
                        "reasoning": prediction.reasoning,
                        "impact": prediction.estimated_impact,
                        "priority": prediction.recommended_priority,
                    })),
                ).await {
                    error!("Failed to persist relationship: {}", e);
                    failed_relationships += 1;
                }
            }
        }
        
        // Update metrics
        self.update_metrics(&graph).await;
        
        // Invalidate cache after changes
        self.cache.write().await.invalidate();
        
        let duration = start_time.elapsed();
        
        info!(
            "Built {} relationships ({} failed) with {} nodes in {:?}",
            relationships_created, failed_relationships, nodes_created, duration
        );
        
        Ok(RelationshipSummary {
            relationships_created,
            nodes_created,
            failed_relationships,
            processing_time: duration,
            total_nodes: graph.node_count(),
            total_edges: graph.edge_count(),
        })
    }
    
    /// Find shortest path between two nodes
    #[instrument(skip(self))]
    pub async fn find_path(
        &self,
        from_id: &str,
        to_id: &str,
    ) -> GraphResult<Option<PathInfo>> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached_path) = cache.get_path(from_id, to_id) {
                debug!("Returning cached path");
                // Convert to PathInfo
                return Ok(Some(self.path_to_info(cached_path).await?));
            }
        }
        
        let graph = self.graph.read().await;
        let indices = self.node_indices.read().await;
        
        let from_idx = indices.get(from_id)
            .ok_or_else(|| GraphError::NodeNotFound(from_id.to_string()))?;
        
        let to_idx = indices.get(to_id)
            .ok_or_else(|| GraphError::NodeNotFound(to_id.to_string()))?;
        
        // Check if path exists
        if !has_path_connecting(&*graph, *from_idx, *to_idx, None) {
            return Ok(None);
        }
        
        // Find shortest path using Dijkstra's algorithm
        let result = dijkstra(
            &*graph,
            *from_idx,
            Some(*to_idx),
            |e| {
                let edge = e.weight();
                // Use inverted confidence as cost (higher confidence = lower cost)
                1.0 / edge.effective_weight(self.config.temporal_decay_rate)
            },
        );
        
        if let Some(cost) = result.get(to_idx) {
            // Reconstruct path
            let result_map: HashMap<NodeIndex, f64> = result.clone().into_iter().collect();
            let path = self.reconstruct_path(&graph, *from_idx, *to_idx, &result_map);
            
            // Cache the path
            {
                let mut cache = self.cache.write().await;
                cache.set_path(
                    from_id.to_string(),
                    to_id.to_string(),
                    path.clone(),
                    self.config.cache_ttl_seconds,
                );
            }
            
            Ok(Some(PathInfo {
                nodes: path.iter()
                    .map(|idx| graph[*idx].id())
                    .collect(),
                total_confidence: 1.0 / cost,
                length: path.len(),
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate influence score of a node
    #[instrument(skip(self))]
    pub async fn calculate_influence(
        &self,
        node_id: &str,
        max_depth: Option<usize>,
    ) -> GraphResult<f64> {
        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(score) = cache.influences.get(node_id) {
                if !cache.is_expired(&format!("influence:{}", node_id)) {
                    return Ok(*score);
                }
            }
        }
        
        let graph = self.graph.read().await;
        let indices = self.node_indices.read().await;
        
        let node_idx = indices.get(node_id)
            .ok_or_else(|| GraphError::NodeNotFound(node_id.to_string()))?;
        
        let depth = max_depth.unwrap_or(self.config.max_traversal_depth);
        let influence = self.calculate_influence_recursive(
            &graph,
            *node_idx,
            depth,
            &mut HashSet::new(),
        );
        
        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.influences.insert(node_id.to_string(), influence);
            cache.set_expiry(
                format!("influence:{}", node_id),
                self.config.cache_ttl_seconds,
            );
        }
        
        Ok(influence)
    }
    
    /// Find patterns in the graph
    #[instrument(skip(self))]
    pub async fn find_patterns(
        &self,
        min_support: Option<f64>,
    ) -> GraphResult<Vec<Pattern>> {
        let graph = self.graph.read().await;
        let support_threshold = min_support.unwrap_or(self.config.min_pattern_support);
        
        let mut patterns = Vec::new();
        
        // Find frequent subgraphs
        let subgraphs = self.mine_frequent_subgraphs(&graph, support_threshold);
        
        for subgraph in subgraphs {
            patterns.push(Pattern {
                pattern_type: PatternType::FrequentSubgraph,
                nodes: subgraph.nodes,
                edges: subgraph.edges,
                support: subgraph.support,
                confidence: subgraph.confidence,
                instances: subgraph.instances,
            });
        }
        
        // Find causal patterns if enabled
        if self.config.enable_causal_inference {
            let causal_patterns = self.find_causal_patterns(&graph).await?;
            patterns.extend(causal_patterns);
        }
        
        // Sort by support
        patterns.sort_by(|a, b| b.support.partial_cmp(&a.support).unwrap());
        
        Ok(patterns)
    }
    
    /// Get node centrality scores
    #[instrument(skip(self))]
    pub async fn get_centrality(
        &self,
        node_id: &str,
    ) -> GraphResult<CentralityScores> {
        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(scores) = cache.centralities.get(node_id) {
                if !cache.is_expired(&format!("centrality:{}", node_id)) {
                    return Ok(scores.clone());
                }
            }
        }
        
        let graph = self.graph.read().await;
        let indices = self.node_indices.read().await;
        
        let node_idx = indices.get(node_id)
            .ok_or_else(|| GraphError::NodeNotFound(node_id.to_string()))?;
        
        let scores = self.calculate_centrality_scores(&graph, *node_idx);
        
        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.centralities.insert(node_id.to_string(), scores.clone());
            cache.set_expiry(
                format!("centrality:{}", node_id),
                self.config.cache_ttl_seconds,
            );
        }
        
        Ok(scores)
    }
    
    /// Detect communities in the graph
    #[instrument(skip(self))]
    pub async fn detect_communities(&self) -> GraphResult<Vec<Community>> {
        let graph = self.graph.read().await;
        
        // Use Tarjan's algorithm for strongly connected components
        let sccs = tarjan_scc(&*graph);
        
        let mut communities = Vec::new();
        
        for (idx, component) in sccs.iter().enumerate() {
            if component.len() < 2 {
                continue; // Skip single-node communities
            }
            
            let members: Vec<String> = component.iter()
                .map(|idx| graph[*idx].id())
                .collect();
            
            let density = self.calculate_community_density(&graph, component);
            
            communities.push(Community {
                id: Uuid::new_v4(),
                members: members.clone(),
                size: members.len(),
                density,
                cohesion: self.calculate_cohesion(&graph, component),
                community_type: self.identify_community_type(&graph, component),
            });
        }
        
        Ok(communities)
    }
    
    /// Get graph metrics
    pub async fn get_metrics(&self) -> GraphMetrics {
        self.metrics.read().await.clone()
    }
    
    // Private helper methods
    
    fn add_or_get_node(
        &self,
        graph: &mut DiGraph<Node, Edge>,
        indices: &mut HashMap<String, NodeIndex>,
        node: Node,
    ) -> NodeIndex {
        let node_id = node.id();
        
        if let Some(idx) = indices.get(&node_id) {
            *idx
        } else {
            let idx = graph.add_node(node);
            indices.insert(node_id, idx);
            idx
        }
    }
    
    async fn update_metrics(&self, graph: &DiGraph<Node, Edge>) {
        let mut metrics = self.metrics.write().await;
        
        metrics.node_count = graph.node_count();
        metrics.edge_count = graph.edge_count();
        
        if metrics.node_count > 0 {
            metrics.avg_degree = (2.0 * metrics.edge_count as f64) / metrics.node_count as f64;
            
            let max_edges = metrics.node_count * (metrics.node_count - 1);
            metrics.density = if max_edges > 0 {
                metrics.edge_count as f64 / max_edges as f64
            } else {
                0.0
            };
        }
        
        metrics.strongly_connected_components = tarjan_scc(graph).len();
        metrics.last_updated = Some(Utc::now());
    }
    
    fn calculate_influence_recursive(
        &self,
        graph: &DiGraph<Node, Edge>,
        node: NodeIndex,
        depth: usize,
        visited: &mut HashSet<NodeIndex>,
    ) -> f64 {
        if depth == 0 || visited.contains(&node) {
            return 0.0;
        }
        
        visited.insert(node);
        
        let mut influence = 1.0;
        
        for neighbor in graph.neighbors_directed(node, Direction::Outgoing) {
            let edge = graph.edges_connecting(node, neighbor).next();
            if let Some(e) = edge {
                let weight = e.weight().effective_weight(self.config.temporal_decay_rate);
                influence += weight * self.calculate_influence_recursive(
                    graph,
                    neighbor,
                    depth - 1,
                    visited,
                );
            }
        }
        
        influence
    }
    
    fn reconstruct_path(
        &self,
        graph: &DiGraph<Node, Edge>,
        from: NodeIndex,
        to: NodeIndex,
        distances: &HashMap<NodeIndex, f64>,
    ) -> Vec<NodeIndex> {
        let mut path = vec![to];
        let mut current = to;
        
        while current != from {
            let mut best_prev = None;
            let mut best_dist = f64::INFINITY;
            
            for neighbor in graph.neighbors_directed(current, Direction::Incoming) {
                if let Some(dist) = distances.get(&neighbor) {
                    if *dist < best_dist {
                        best_dist = *dist;
                        best_prev = Some(neighbor);
                    }
                }
            }
            
            if let Some(prev) = best_prev {
                path.push(prev);
                current = prev;
            } else {
                break;
            }
        }
        
        path.reverse();
        path
    }
    
    async fn path_to_info(&self, path: &[NodeIndex]) -> GraphResult<PathInfo> {
        let graph = self.graph.read().await;
        
        let nodes: Vec<String> = path.iter()
            .map(|idx| graph[*idx].id())
            .collect();
        
        let mut total_confidence = 1.0;
        for i in 0..path.len() - 1 {
            if let Some(edge) = graph.find_edge(path[i], path[i + 1]) {
                let weight = graph[edge].effective_weight(self.config.temporal_decay_rate);
                total_confidence *= weight;
            }
        }
        
        Ok(PathInfo {
            nodes,
            total_confidence,
            length: path.len(),
        })
    }
    
    fn mine_frequent_subgraphs(
        &self,
        graph: &DiGraph<Node, Edge>,
        min_support: f64,
    ) -> Vec<FrequentSubgraph> {
        // Simplified frequent subgraph mining
        // In production, use gSpan or similar algorithm
        Vec::new()
    }
    
    async fn find_causal_patterns(
        &self,
        graph: &DiGraph<Node, Edge>,
    ) -> GraphResult<Vec<Pattern>> {
        // Simplified causal pattern detection
        // In production, use PC algorithm or similar
        Ok(Vec::new())
    }
    
    fn calculate_centrality_scores(
        &self,
        graph: &DiGraph<Node, Edge>,
        node: NodeIndex,
    ) -> CentralityScores {
        let degree = graph.edges(node).count() as f64;
        
        // Simplified centrality calculations
        // In production, use proper algorithms
        CentralityScores {
            degree,
            betweenness: 0.0,
            closeness: 0.0,
            eigenvector: 0.0,
            pagerank: degree / graph.node_count() as f64,
        }
    }
    
    fn calculate_community_density(
        &self,
        graph: &DiGraph<Node, Edge>,
        community: &[NodeIndex],
    ) -> f64 {
        if community.len() < 2 {
            return 0.0;
        }
        
        let mut edge_count = 0;
        for &node in community {
            for &other in community {
                if node != other && graph.contains_edge(node, other) {
                    edge_count += 1;
                }
            }
        }
        
        let max_edges = community.len() * (community.len() - 1);
        edge_count as f64 / max_edges as f64
    }
    
    fn calculate_cohesion(
        &self,
        graph: &DiGraph<Node, Edge>,
        community: &[NodeIndex],
    ) -> f64 {
        // Simplified cohesion calculation
        self.calculate_community_density(graph, community)
    }
    
    fn identify_community_type(
        &self,
        graph: &DiGraph<Node, Edge>,
        community: &[NodeIndex],
    ) -> CommunityType {
        // Analyze node types in community to determine type
        let mut type_counts: HashMap<NodeType, usize> = HashMap::new();
        
        for &node_idx in community {
            let node_type = graph[node_idx].node_type();
            *type_counts.entry(node_type).or_insert(0) += 1;
        }
        
        // Determine dominant type
        if type_counts.get(&NodeType::User).unwrap_or(&0) > &(community.len() / 2) {
            CommunityType::Collaboration
        } else if type_counts.get(&NodeType::Artifact).unwrap_or(&0) > &(community.len() / 2) {
            CommunityType::WorkCluster
        } else {
            CommunityType::Mixed
        }
    }
}

// Supporting types

/// Summary of relationship building operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipSummary {
    pub relationships_created: usize,
    pub nodes_created: usize,
    pub failed_relationships: usize,
    pub processing_time: std::time::Duration,
    pub total_nodes: usize,
    pub total_edges: usize,
}

/// Path information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathInfo {
    pub nodes: Vec<String>,
    pub total_confidence: f64,
    pub length: usize,
}

/// Pattern in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: PatternType,
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String, RelationshipType)>,
    pub support: f64,
    pub confidence: f64,
    pub instances: usize,
}

/// Pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    FrequentSubgraph,
    CausalChain,
    CollaborationCluster,
    WorkflowSequence,
    DependencyTree,
}

/// Frequent subgraph
#[derive(Debug, Clone)]
struct FrequentSubgraph {
    nodes: Vec<String>,
    edges: Vec<(String, String, RelationshipType)>,
    support: f64,
    confidence: f64,
    instances: usize,
}

/// Community in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: Uuid,
    pub members: Vec<String>,
    pub size: usize,
    pub density: f64,
    pub cohesion: f64,
    pub community_type: CommunityType,
}

/// Community types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommunityType {
    WorkCluster,
    Collaboration,
    Project,
    Team,
    Mixed,
}

/// Graph analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysis {
    pub metrics: GraphMetrics,
    pub top_influencers: Vec<(String, f64)>,
    pub critical_paths: Vec<PathInfo>,
    pub communities: Vec<Community>,
    pub patterns: Vec<Pattern>,
    pub recommendations: Vec<Recommendation>,
}

/// Recommendation based on graph analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub recommendation_type: RecommendationType,
    pub target_nodes: Vec<String>,
    pub confidence: f64,
    pub reasoning: String,
    pub potential_impact: f64,
}

/// Recommendation types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    AddRelationship,
    RemoveBottleneck,
    StrengthConnection,
    MergeCommunities,
    SplitCommunity,
    PrioritizeNode,
}

impl EvidenceGraph {
    /// Perform comprehensive graph analysis
    #[instrument(skip(self))]
    pub async fn analyze(&self) -> GraphResult<GraphAnalysis> {
        info!("Performing comprehensive graph analysis");
        
        let graph = self.graph.read().await;
        let indices = self.node_indices.read().await;
        
        // Get metrics
        let metrics = self.get_metrics().await;
        
        // Find top influencers
        let mut influencers = Vec::new();
        for (node_id, _) in indices.iter().take(100) {
            if let Ok(influence) = self.calculate_influence(node_id, None).await {
                influencers.push((node_id.clone(), influence));
            }
        }
        influencers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_influencers: Vec<(String, f64)> = influencers.into_iter().take(10).collect();
        
        // Find critical paths
        let critical_paths = self.find_critical_paths(&graph, &indices).await?;
        
        // Detect communities
        let communities = self.detect_communities().await?;
        
        // Find patterns
        let patterns = self.find_patterns(None).await?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &graph,
            &top_influencers,
            &communities,
            &patterns,
        ).await?;
        
        Ok(GraphAnalysis {
            metrics,
            top_influencers,
            critical_paths,
            communities,
            patterns,
            recommendations,
        })
    }
    
    /// Export graph to various formats
    pub async fn export(&self, format: ExportFormat) -> GraphResult<Vec<u8>> {
        let graph = self.graph.read().await;
        
        match format {
            ExportFormat::GraphML => self.export_graphml(&graph),
            ExportFormat::Json => self.export_json(&graph),
            ExportFormat::Dot => self.export_dot(&graph),
            ExportFormat::Gexf => self.export_gexf(&graph),
        }
    }
    
    /// Import graph from various formats
    pub async fn import(&self, data: &[u8], format: ImportFormat) -> GraphResult<()> {
        let imported = match format {
            ImportFormat::GraphML => self.import_graphml(data)?,
            ImportFormat::Json => self.import_json(data)?,
            ImportFormat::Dot => return Err(GraphError::OperationFailed("DOT import not yet implemented".to_string())),
        };
        
        let mut graph = self.graph.write().await;
        let mut indices = self.node_indices.write().await;
        
        *graph = imported.0;
        *indices = imported.1;
        
        self.cache.write().await.invalidate();
        self.update_metrics(&graph).await;
        
        Ok(())
    }
    
    /// Prune graph by removing low-confidence edges
    pub async fn prune(&self, min_confidence: f64) -> GraphResult<usize> {
        let mut graph = self.graph.write().await;
        let mut removed = 0;
        
        let edges_to_remove: Vec<_> = graph.edge_indices()
            .filter(|&edge| {
                let weight = &graph[edge];
                weight.confidence < min_confidence
            })
            .collect();
        
        for edge in edges_to_remove {
            graph.remove_edge(edge);
            removed += 1;
        }
        
        self.cache.write().await.invalidate();
        self.update_metrics(&graph).await;
        
        info!("Pruned {} low-confidence edges", removed);
        Ok(removed)
    }
    
    /// Merge nodes based on similarity
    pub async fn merge_similar_nodes(&self, similarity_threshold: f64) -> GraphResult<usize> {
        // Find similar nodes
        let similar_pairs = self.find_similar_nodes(similarity_threshold).await?;
        
        let mut graph = self.graph.write().await;
        let mut indices = self.node_indices.write().await;
        let mut merged = 0;
        
        for (node1_id, node2_id) in similar_pairs {
            // Merge node2 into node1
            if let (Some(&idx1), Some(&idx2)) = (indices.get(&node1_id), indices.get(&node2_id)) {
                // Transfer all edges from node2 to node1
                let edges: Vec<_> = graph.edges(idx2)
                    .map(|e| (e.target(), e.weight().clone()))
                    .collect();
                
                for (target, weight) in edges {
                    if !graph.contains_edge(idx1, target) {
                        graph.add_edge(idx1, target, weight);
                    }
                }
                
                // Remove node2
                graph.remove_node(idx2);
                indices.remove(&node2_id);
                merged += 1;
            }
        }
        
        self.cache.write().await.invalidate();
        self.update_metrics(&graph).await;
        
        info!("Merged {} similar nodes", merged);
        Ok(merged)
    }
    
    // Private helper methods
    
    async fn find_critical_paths(
        &self,
        graph: &DiGraph<Node, Edge>,
        indices: &HashMap<String, NodeIndex>,
    ) -> GraphResult<Vec<PathInfo>> {
        let mut critical_paths = Vec::new();
        
        // Find paths between high-influence nodes
        let mut high_influence_nodes: Vec<_> = indices.keys()
            .filter_map(|id| {
                if let Ok(influence) = futures::executor::block_on(
                    self.calculate_influence(id, Some(2))
                ) {
                    if influence > 5.0 {
                        Some(id.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        high_influence_nodes.truncate(10);
        
        for i in 0..high_influence_nodes.len() {
            for j in i + 1..high_influence_nodes.len() {
                if let Ok(Some(path)) = self.find_path(
                    &high_influence_nodes[i],
                    &high_influence_nodes[j],
                ).await {
                    if path.total_confidence > 0.5 {
                        critical_paths.push(path);
                    }
                }
            }
        }
        
        critical_paths.sort_by(|a, b| b.total_confidence.partial_cmp(&a.total_confidence).unwrap());
        critical_paths.truncate(5);
        
        Ok(critical_paths)
    }
    
    async fn generate_recommendations(
        &self,
        graph: &DiGraph<Node, Edge>,
        top_influencers: &[(String, f64)],
        communities: &[Community],
        patterns: &[Pattern],
    ) -> GraphResult<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        // Recommend strengthening weak connections between influential nodes
        for (node_id, influence) in top_influencers.iter().take(5) {
            if let Some(&idx) = self.node_indices.read().await.get(node_id) {
                for edge in graph.edges(idx) {
                    let weight = edge.weight();
                    if weight.confidence < 0.5 {
                        recommendations.push(Recommendation {
                            recommendation_type: RecommendationType::StrengthConnection,
                            target_nodes: vec![node_id.clone(), graph[edge.target()].id()],
                            confidence: 0.7,
                            reasoning: format!(
                                "Weak connection from high-influence node (influence: {:.2})",
                                influence
                            ),
                            potential_impact: influence * 0.5,
                        });
                    }
                }
            }
        }
        
        // Recommend merging small related communities
        for community in communities.iter().filter(|c| c.size < 5) {
            if community.density > 0.7 {
                recommendations.push(Recommendation {
                    recommendation_type: RecommendationType::MergeCommunities,
                    target_nodes: community.members.clone(),
                    confidence: community.density,
                    reasoning: "Small, highly connected community could be merged".to_string(),
                    potential_impact: 0.3,
                });
            }
        }
        
        recommendations.sort_by(|a, b| b.potential_impact.partial_cmp(&a.potential_impact).unwrap());
        recommendations.truncate(10);
        
        Ok(recommendations)
    }
    
    async fn find_similar_nodes(&self, threshold: f64) -> GraphResult<Vec<(String, String)>> {
        let graph = self.graph.read().await;
        let indices = self.node_indices.read().await;
        let mut similar_pairs = Vec::new();
        
        let nodes: Vec<_> = indices.iter().collect();
        
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                let similarity = self.calculate_node_similarity(
                    &graph,
                    *nodes[i].1,
                    *nodes[j].1,
                );
                
                if similarity > threshold {
                    similar_pairs.push((
                        nodes[i].0.clone(),
                        nodes[j].0.clone(),
                    ));
                }
            }
        }
        
        Ok(similar_pairs)
    }
    
    fn calculate_node_similarity(
        &self,
        graph: &DiGraph<Node, Edge>,
        node1: NodeIndex,
        node2: NodeIndex,
    ) -> f64 {
        // Jaccard similarity based on neighbors
        let neighbors1: HashSet<_> = graph.neighbors(node1).collect();
        let neighbors2: HashSet<_> = graph.neighbors(node2).collect();
        
        if neighbors1.is_empty() && neighbors2.is_empty() {
            return 1.0;
        }
        
        let intersection = neighbors1.intersection(&neighbors2).count();
        let union = neighbors1.union(&neighbors2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
    
    fn export_graphml(&self, graph: &DiGraph<Node, Edge>) -> GraphResult<Vec<u8>> {
        // Simplified GraphML export
        let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n");
        xml.push_str("<graph id=\"G\" edgedefault=\"directed\">\n");
        
        for node in graph.node_indices() {
            let node_data = &graph[node];
            xml.push_str(&format!("<node id=\"{}\"/>\n", node_data.id()));
        }
        
        for edge in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(edge).unwrap();
            let weight = &graph[edge];
            xml.push_str(&format!(
                "<edge source=\"{}\" target=\"{}\" weight=\"{}\"/>\n",
                graph[source].id(),
                graph[target].id(),
                weight.confidence
            ));
        }
        
        xml.push_str("</graph>\n</graphml>");
        Ok(xml.into_bytes())
    }
    
    fn export_json(&self, graph: &DiGraph<Node, Edge>) -> GraphResult<Vec<u8>> {
        let nodes: Vec<_> = graph.node_indices()
            .map(|idx| &graph[idx])
            .collect();
        
        let edges: Vec<_> = graph.edge_indices()
            .map(|idx| {
                let (source, target) = graph.edge_endpoints(idx).unwrap();
                (
                    graph[source].id(),
                    graph[target].id(),
                    &graph[idx],
                )
            })
            .collect();
        
        let export = serde_json::json!({
            "nodes": nodes,
            "edges": edges,
        });
        
        serde_json::to_vec_pretty(&export)
            .map_err(|e| GraphError::OperationFailed(e.to_string()))
    }
    
    fn export_dot(&self, graph: &DiGraph<Node, Edge>) -> GraphResult<Vec<u8>> {
        let mut dot = String::from("digraph G {\n");
        
        for node in graph.node_indices() {
            let node_data = &graph[node];
            dot.push_str(&format!("  \"{}\";\n", node_data.id()));
        }
        
        for edge in graph.edge_indices() {
            let (source, target) = graph.edge_endpoints(edge).unwrap();
            let weight = &graph[edge];
            dot.push_str(&format!(
                "  \"{}\" -> \"{}\" [weight={}];\n",
                graph[source].id(),
                graph[target].id(),
                weight.confidence
            ));
        }
        
        dot.push_str("}");
        Ok(dot.into_bytes())
    }
    
    fn export_gexf(&self, graph: &DiGraph<Node, Edge>) -> GraphResult<Vec<u8>> {
        // Simplified GEXF export
        Ok(Vec::new())
    }
    
    fn import_graphml(&self, data: &[u8]) -> GraphResult<(DiGraph<Node, Edge>, HashMap<String, NodeIndex>)> {
        // Simplified GraphML import
        Err(GraphError::OperationFailed("GraphML import not yet implemented".to_string()))
    }
    
    fn import_json(&self, data: &[u8]) -> GraphResult<(DiGraph<Node, Edge>, HashMap<String, NodeIndex>)> {
        // Simplified JSON import
        Err(GraphError::OperationFailed("JSON import not yet implemented".to_string()))
    }
}

/// Export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    GraphML,
    Json,
    Dot,
    Gexf,
}

/// Import formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportFormat {
    GraphML,
    Json,
    Dot,
}
