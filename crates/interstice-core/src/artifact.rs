use crate::{Platform, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    PullRequest { number: u32, repo: Option<String> },
    Issue { id: String, project: Option<String> },
    Commit { sha: String },
    Document { title: String, url: Option<String> },
    Message { content: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub artifact_type: ArtifactType,
    pub platform: Platform,
    pub raw_text: String,
    pub metadata: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct ArtifactExtractor {
    pr_regex: Regex,
    issue_regex: Regex,
    commit_regex: Regex,
}

impl ArtifactExtractor {
    pub fn new() -> Self {
        Self {
            pr_regex: Regex::new(r"(?i)(?:PR|pull request|MR|merge request)\s*#?(\d+)").unwrap(),
            issue_regex: Regex::new(r"([A-Z]{2,}-\d+)").unwrap(),
            commit_regex: Regex::new(r"(?i)(?:commit|sha)\s*([a-f0-9]{7,40})").unwrap(),
        }
    }

    pub async fn extract(&self, content: &str, platform: Platform) -> Result<Vec<Artifact>> {
        let mut artifacts = Vec::new();
        let now = chrono::Utc::now();

        // Extract PRs
        for cap in self.pr_regex.captures_iter(content) {
            artifacts.push(Artifact {
                id: format!("{}-pr-{}", platform, &cap[1]),
                artifact_type: ArtifactType::PullRequest {
                    number: cap[1].parse().unwrap(),
                    repo: None,
                },
                platform,
                raw_text: cap[0].to_string(),
                metadata: serde_json::json!({}),
                timestamp: now,
            });
        }

        // Extract Issues/Tickets
        for cap in self.issue_regex.captures_iter(content) {
            artifacts.push(Artifact {
                id: format!("{}-issue-{}", platform, &cap[1]),
                artifact_type: ArtifactType::Issue {
                    id: cap[1].to_string(),
                    project: None,
                },
                platform,
                raw_text: cap[0].to_string(),
                metadata: serde_json::json!({}),
                timestamp: now,
            });
        }

        // Extract Commits
        for cap in self.commit_regex.captures_iter(content) {
            artifacts.push(Artifact {
                id: format!("{}-commit-{}", platform, &cap[1]),
                artifact_type: ArtifactType::Commit {
                    sha: cap[1].to_string(),
                },
                platform,
                raw_text: cap[0].to_string(),
                metadata: serde_json::json!({}),
                timestamp: now,
            });
        }

        // If no specific artifacts found, treat the whole content as a message
        if artifacts.is_empty() && !content.trim().is_empty() {
            artifacts.push(Artifact {
                id: format!("{}-msg-{}", platform, uuid::Uuid::new_v4()),
                artifact_type: ArtifactType::Message {
                    content: content.to_string(),
                },
                platform,
                raw_text: content.to_string(),
                metadata: serde_json::json!({}),
                timestamp: now,
            });
        }

        Ok(artifacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_extract_pr() {
        let extractor = ArtifactExtractor::new();
        let content = "Just merged PR #123 to fix the bug";
        let artifacts = extractor.extract(content, Platform::GitHub).await.unwrap();

        assert_eq!(artifacts.len(), 1);
        match &artifacts[0].artifact_type {
            ArtifactType::PullRequest { number, .. } => assert_eq!(*number, 123),
            _ => panic!("Expected PullRequest"),
        }
    }

    #[tokio::test]
    async fn test_extract_jira_ticket() {
        let extractor = ArtifactExtractor::new();
        let content = "Working on PROJ-456 today";
        let artifacts = extractor.extract(content, Platform::Jira).await.unwrap();

        assert_eq!(artifacts.len(), 1);
        match &artifacts[0].artifact_type {
            ArtifactType::Issue { id, .. } => assert_eq!(id, "PROJ-456"),
            _ => panic!("Expected Issue"),
        }
    }
}
