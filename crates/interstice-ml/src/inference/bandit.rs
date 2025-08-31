use anyhow::Result;
use rand_distr::{Distribution, Gamma};
use rand::rng;

pub struct ThompsonSamplingBandit {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    n_arms: usize,
}

impl ThompsonSamplingBandit {
    pub fn new(n_arms: usize) -> Self {
        Self {
            alpha: vec![1.0; n_arms],
            beta: vec![1.0; n_arms],
            n_arms,
        }
    }

    pub fn sample(&self, base_predictions: Vec<f32>) -> Result<Vec<f32>> {
        let mut rng = rng();
        let mut samples = Vec::new();
        
        for i in 0..self.n_arms {
            let alpha_dist = Gamma::new(self.alpha[i], 1.0)
                .map_err(|e| anyhow::anyhow!("Failed to create Gamma distribution: {}", e))?;
            let beta_dist = Gamma::new(self.beta[i], 1.0)
                .map_err(|e| anyhow::anyhow!("Failed to create Gamma distribution: {}", e))?;
            
            let alpha_sample = alpha_dist.sample(&mut rng);
            let beta_sample = beta_dist.sample(&mut rng);
            
            let theta = alpha_sample / (alpha_sample + beta_sample);
            
            let combined = if i < base_predictions.len() {
                theta * 0.3 + base_predictions[i] * 0.7
            } else {
                theta
            };
            
            samples.push(combined);
        }
        
        Ok(samples)
    }

    pub fn update(&mut self, arm: usize, reward: f32) {
        if arm < self.n_arms {
            if reward > 0.5 {
                self.alpha[arm] += reward;
            } else {
                self.beta[arm] += 1.0 - reward;
            }
        }
    }
}