// src/handlers/slack/constants.rs

use ring::aead::LessSafeKey;
use std::sync::OnceLock;

pub const SLACK_OAUTH_URL: &str = "https://slack.com/api/oauth.v2.access";
pub const SIGNATURE_VERSION: &str = "v0";
pub const MAX_TIMESTAMP_AGE_SECS: i64 = 300; // 5 minutes
pub const EXPECTED_TOKEN_TYPE: &str = "Bearer"; // OAuth 2.0 standard
pub static ENCRYPTION_KEY: OnceLock<LessSafeKey> = OnceLock::new();
pub const MAX_BODY_SIZE: usize = 256 * 1024;