//interstice-ml/src/inference/bandit.rs
use std::fmt;
use std::sync::Arc;

use anyhow::{Result};
use rand::rngs::StdRng;
use rand::{SeedableRng};
use rand_distr::{Distribution, Gamma};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Custom error types for the bandit module
#[derive(Error, Debug)]
pub enum BanditError {
    #[error("Invalid arm index {arm}: must be less than {n_arms}")]
    InvalidArm { arm: usize, n_arms: usize },
    
    #[error("Invalid reward value {reward}: must be in range [0, 1]")]
    InvalidReward { reward: f32 },
    
    #[error("Mismatched dimensions: expected {expected} predictions, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Distribution creation failed: {0}")]
    DistributionError(String),
}

/// Configuration for the Thompson Sampling bandit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditConfig {
    /// Number of arms in the bandit
    pub n_arms: usize,
    
    /// Initial alpha parameter for Beta distribution (successes + 1)
    pub initial_alpha: f32,
    
    /// Initial beta parameter for Beta distribution (failures + 1)
    pub initial_beta: f32,
    
    /// Weight for combining sampled values with base predictions (0.0 to 1.0)
    /// Higher values give more weight to base predictions
    pub base_prediction_weight: f32,
    
    /// Optional seed for reproducible randomness
    pub seed: Option<u64>,
    
    /// Enable adaptive exploration decay
    pub adaptive_exploration: bool,
    
    /// Minimum exploration rate when adaptive exploration is enabled
    pub min_exploration_rate: f32,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            n_arms: 10,
            initial_alpha: 1.0,
            initial_beta: 1.0,
            base_prediction_weight: 0.7,
            seed: None,
            adaptive_exploration: false,
            min_exploration_rate: 0.1,
        }
    }
}

impl BanditConfig {
    /// Validates the configuration parameters
    pub fn validate(&self) -> Result<(), BanditError> {
        if self.n_arms == 0 {
            return Err(BanditError::InvalidParameter(
                "n_arms must be greater than 0".to_string()
            ));
        }
        
        if self.initial_alpha <= 0.0 || self.initial_beta <= 0.0 {
            return Err(BanditError::InvalidParameter(
                "Alpha and beta parameters must be positive".to_string()
            ));
        }
        
        if !(0.0..=1.0).contains(&self.base_prediction_weight) {
            return Err(BanditError::InvalidParameter(
                "base_prediction_weight must be between 0 and 1".to_string()
            ));
        }
        
        if !(0.0..=1.0).contains(&self.min_exploration_rate) {
            return Err(BanditError::InvalidParameter(
                "min_exploration_rate must be between 0 and 1".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Statistics for each arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStatistics {
    pub alpha: f32,
    pub beta: f32,
    pub total_pulls: u64,
    pub total_reward: f64,
    pub mean_reward: f64,
    pub last_reward: Option<f32>,
}

impl ArmStatistics {
    fn new(initial_alpha: f32, initial_beta: f32) -> Self {
        Self {
            alpha: initial_alpha,
            beta: initial_beta,
            total_pulls: 0,
            total_reward: 0.0,
            mean_reward: 0.0,
            last_reward: None,
        }
    }
    
    fn update(&mut self, reward: f32) {
        self.total_pulls += 1;
        self.total_reward += reward as f64;
        self.mean_reward = self.total_reward / self.total_pulls as f64;
        self.last_reward = Some(reward);
        
        // Update Beta distribution parameters
        // Using a more nuanced update strategy
        if reward > 0.5 {
            self.alpha += reward;
        } else {
            self.beta += 1.0 - reward;
        }
    }
}

/// Thompson Sampling Multi-Armed Bandit
/// 
/// This implementation uses Beta distributions to model the uncertainty
/// about each arm's reward probability and samples from these distributions
/// to balance exploration and exploitation.
#[derive(Debug, Clone)]
pub struct ThompsonSamplingBandit {
    config: BanditConfig,
    arms: Vec<ArmStatistics>,
    rng: Arc<parking_lot::Mutex<StdRng>>,
    total_updates: u64,
}

impl ThompsonSamplingBandit {
    /// Creates a new Thompson Sampling bandit with the given configuration
    pub fn new(config: BanditConfig) -> Result<Self> {
        config.validate()?;
        
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random::<u64>()),
        };
        
        let arms = (0..config.n_arms)
            .map(|_| ArmStatistics::new(config.initial_alpha, config.initial_beta))
            .collect();
        
        Ok(Self {
            config,
            arms,
            rng: Arc::new(parking_lot::Mutex::new(rng)),
            total_updates: 0,
        })
    }
    
    /// Creates a new bandit with default configuration
    pub fn with_arms(n_arms: usize) -> Result<Self> {
        let config = BanditConfig {
            n_arms,
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Samples from the bandit, optionally combining with base predictions
    /// 
    /// # Arguments
    /// * `base_predictions` - Optional base predictions to combine with Thompson samples
    /// 
    /// # Returns
    /// A vector of scores for each arm
    pub fn sample(&self, base_predictions: Option<&[f32]>) -> Result<Vec<f32>> {
        // Validate base predictions if provided
        if let Some(preds) = base_predictions {
            if preds.len() != self.config.n_arms {
                return Err(BanditError::DimensionMismatch {
                    expected: self.config.n_arms,
                    actual: preds.len(),
                }.into());
            }
            
            // Validate prediction values
            for (i, &pred) in preds.iter().enumerate() {
                if !pred.is_finite() {
                    return Err(BanditError::InvalidParameter(
                        format!("Base prediction at index {} is not finite: {}", i, pred)
                    ).into());
                }
            }
        }
        
        let mut rng = self.rng.lock();
        let mut samples = Vec::with_capacity(self.config.n_arms);
        
        // Calculate exploration rate for adaptive exploration
        let exploration_weight = if self.config.adaptive_exploration {
            self.calculate_exploration_weight()
        } else {
            1.0 - self.config.base_prediction_weight
        };
        
        for (i, arm_stats) in self.arms.iter().enumerate() {
            let theta = self.sample_theta(arm_stats, &mut *rng)?;
            
            let combined_score = match base_predictions {
                Some(preds) => {
                    // Combine Thompson sample with base prediction
                    theta * exploration_weight + preds[i] * (1.0 - exploration_weight)
                }
                None => theta,
            };
            
            samples.push(combined_score.clamp(0.0, 1.0));
        }
        
        Ok(samples)
    }
    
    /// Samples a single theta value from the Beta distribution for an arm
    fn sample_theta(&self, arm_stats: &ArmStatistics, rng: &mut StdRng) -> Result<f32> {
        // Use Gamma distribution to sample from Beta
        // Beta(α, β) can be sampled as X/(X+Y) where X ~ Gamma(α, 1), Y ~ Gamma(β, 1)
        let alpha_dist = Gamma::new(arm_stats.alpha, 1.0)
            .map_err(|e| BanditError::DistributionError(format!("Alpha: {}", e)))?;
        
        let beta_dist = Gamma::new(arm_stats.beta, 1.0)
            .map_err(|e| BanditError::DistributionError(format!("Beta: {}", e)))?;
        
        let alpha_sample = alpha_dist.sample(rng);
        let beta_sample = beta_dist.sample(rng);
        
        Ok(alpha_sample / (alpha_sample + beta_sample))
    }
    
    /// Calculates adaptive exploration weight based on total updates
    fn calculate_exploration_weight(&self) -> f32 {
        // Decay exploration over time
        let decay_rate = 0.999_f32.powf(self.total_updates as f32 / 100.0);
        let exploration = (1.0 - self.config.base_prediction_weight) * decay_rate;
        exploration.max(self.config.min_exploration_rate)
    }
    
    /// Updates the bandit with observed reward for a chosen arm
    /// 
    /// # Arguments
    /// * `arm` - The index of the chosen arm
    /// * `reward` - The observed reward (should be in [0, 1])
    pub fn update(&mut self, arm: usize, reward: f32) -> Result<()> {
        // Validate inputs
        if arm >= self.config.n_arms {
            return Err(BanditError::InvalidArm {
                arm,
                n_arms: self.config.n_arms,
            }.into());
        }
        
        if !(0.0..=1.0).contains(&reward) || !reward.is_finite() {
            return Err(BanditError::InvalidReward { reward }.into());
        }
        
        self.arms[arm].update(reward);
        self.total_updates += 1;
        
        Ok(())
    }
    
    /// Batch update for multiple arm-reward pairs
    pub fn batch_update(&mut self, updates: &[(usize, f32)]) -> Result<()> {
        for &(arm, reward) in updates {
            self.update(arm, reward)?;
        }
        Ok(())
    }
    
    /// Returns the best arm based on current mean rewards
    pub fn best_arm(&self) -> usize {
        self.arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.mean_reward.partial_cmp(&b.mean_reward).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Returns statistics for all arms
    pub fn get_statistics(&self) -> &[ArmStatistics] {
        &self.arms
    }
    
    /// Returns statistics for a specific arm
    pub fn get_arm_statistics(&self, arm: usize) -> Option<&ArmStatistics> {
        self.arms.get(arm)
    }
    
    /// Resets the bandit to initial state
    pub fn reset(&mut self) {
        self.arms = (0..self.config.n_arms)
            .map(|_| ArmStatistics::new(self.config.initial_alpha, self.config.initial_beta))
            .collect();
        self.total_updates = 0;
    }
    
    /// Returns the total number of updates
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }
    
    /// Exports the bandit state for persistence
    pub fn export_state(&self) -> BanditState {
        BanditState {
            config: self.config.clone(),
            arms: self.arms.clone(),
            total_updates: self.total_updates,
        }
    }
    
    /// Restores the bandit from a saved state
    pub fn from_state(state: BanditState) -> Result<Self> {
        state.config.validate()?;
        
        let rng = match state.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random::<u64>()),
        };
        
        Ok(Self {
            config: state.config,
            arms: state.arms,
            rng: Arc::new(parking_lot::Mutex::new(rng)),
            total_updates: state.total_updates,
        })
    }
}

/// Serializable bandit state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditState {
    pub config: BanditConfig,
    pub arms: Vec<ArmStatistics>,
    pub total_updates: u64,
}

impl fmt::Display for ThompsonSamplingBandit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Thompson Sampling Bandit")?;
        writeln!(f, "  Arms: {}", self.config.n_arms)?;
        writeln!(f, "  Total Updates: {}", self.total_updates)?;
        writeln!(f, "  Best Arm: {}", self.best_arm())?;
        
        for (i, arm) in self.arms.iter().enumerate() {
            writeln!(
                f,
                "  Arm {}: α={:.2}, β={:.2}, pulls={}, mean={:.3}",
                i, arm.alpha, arm.beta, arm.total_pulls, arm.mean_reward
            )?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bandit_creation() {
        let config = BanditConfig {
            n_arms: 5,
            seed: Some(42),
            ..Default::default()
        };
        
        let bandit = ThompsonSamplingBandit::new(config).unwrap();
        assert_eq!(bandit.arms.len(), 5);
        assert_eq!(bandit.total_updates(), 0);
    }
    
    #[test]
    fn test_sampling() {
        let config = BanditConfig {
            n_arms: 3,
            seed: Some(42),
            ..Default::default()
        };
        
        let bandit = ThompsonSamplingBandit::new(config).unwrap();
        
        // Sample without base predictions
        let samples = bandit.sample(None).unwrap();
        assert_eq!(samples.len(), 3);
        for sample in &samples {
            assert!((0.0..=1.0).contains(sample));
        }
        
        // Sample with base predictions
        let base_preds = vec![0.2, 0.5, 0.8];
        let samples = bandit.sample(Some(&base_preds)).unwrap();
        assert_eq!(samples.len(), 3);
    }
    
    #[test]
    fn test_update() {
        let mut bandit = ThompsonSamplingBandit::with_arms(3).unwrap();
        
        // Valid update
        bandit.update(0, 0.8).unwrap();
        assert_eq!(bandit.total_updates(), 1);
        
        let stats = bandit.get_arm_statistics(0).unwrap();
        assert_eq!(stats.total_pulls, 1);
        assert_relative_eq!(stats.mean_reward, 0.8, epsilon = 1e-6);
        
        // Invalid arm
        assert!(bandit.update(5, 0.5).is_err());
        
        // Invalid reward
        assert!(bandit.update(0, 1.5).is_err());
    }
    
    #[test]
    fn test_batch_update() {
        let mut bandit = ThompsonSamplingBandit::with_arms(3).unwrap();
        
        let updates = vec![(0, 0.8), (1, 0.3), (0, 0.9), (2, 0.6)];
        bandit.batch_update(&updates).unwrap();
        
        assert_eq!(bandit.total_updates(), 4);
        assert_eq!(bandit.get_arm_statistics(0).unwrap().total_pulls, 2);
    }
    
    #[test]
    fn test_best_arm() {
        let mut bandit = ThompsonSamplingBandit::with_arms(3).unwrap();
        
        // Arm 0: high rewards
        bandit.update(0, 0.9).unwrap();
        bandit.update(0, 0.8).unwrap();
        
        // Arm 1: medium rewards
        bandit.update(1, 0.5).unwrap();
        bandit.update(1, 0.6).unwrap();
        
        // Arm 2: low rewards
        bandit.update(2, 0.2).unwrap();
        bandit.update(2, 0.3).unwrap();
        
        assert_eq!(bandit.best_arm(), 0);
    }
    
    #[test]
    fn test_state_persistence() {
        let mut bandit = ThompsonSamplingBandit::with_arms(2).unwrap();
        bandit.update(0, 0.7).unwrap();
        bandit.update(1, 0.4).unwrap();
        
        let state = bandit.export_state();
        let restored = ThompsonSamplingBandit::from_state(state).unwrap();
        
        assert_eq!(restored.total_updates(), bandit.total_updates());
        assert_eq!(restored.arms.len(), bandit.arms.len());
    }
    
    #[test]
    fn test_adaptive_exploration() {
        let config = BanditConfig {
            n_arms: 3,
            seed: Some(42),
            adaptive_exploration: true,
            min_exploration_rate: 0.05,
            ..Default::default()
        };
        
        let mut bandit = ThompsonSamplingBandit::new(config).unwrap();
        
        // Perform many updates
        for _ in 0..1000 {
            bandit.update(0, 0.6).unwrap();
        }
        
        let weight = bandit.calculate_exploration_weight();
        assert!(weight >= 0.05);
        assert!(weight < 0.3); // Should have decayed
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let bandit = ThompsonSamplingBandit::with_arms(3).unwrap();
        let base_preds = vec![0.5, 0.6]; // Wrong size
        
        let result = bandit.sample(Some(&base_preds));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_config_validation() {
        // Invalid n_arms
        let config = BanditConfig {
            n_arms: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        
        // Invalid alpha
        let config = BanditConfig {
            initial_alpha: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        
        // Invalid weight
        let config = BanditConfig {
            base_prediction_weight: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}