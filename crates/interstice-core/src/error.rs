//! # Error Module
//! 
//! Centralized error handling for the INTERSTICE-ENGINE WorkOS.
//! Provides comprehensive error types with context, recovery suggestions, and tracing.

use std::error::Error as StdError;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::outcome::OutcomeId;
use crate::types::{Platform, UserId, WorkspaceId};

/// Core error type for the system
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Storage error: {0}")]
    Storage(#[source] StorageError),
    
    #[error("Network error: {0}")]
    Network(#[source] NetworkError),
    
    #[error("Authentication error: {0}")]
    Authentication(#[source] AuthError),
    
    #[error("Authorization error: {0}")]
    Authorization(#[source] AuthError),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Not found: {resource_type} with id {resource_id}")]
    NotFound {
        resource_type: String,
        resource_id: String,
    },
    
    #[error("Already exists: {resource_type} with id {resource_id}")]
    AlreadyExists {
        resource_type: String,
        resource_id: String,
    },
    
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Timeout after {0:?}")]
    Timeout(std::time::Duration),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("External service error: {service} - {message}")]
    ExternalService {
        service: String,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("Database error: {0}")]
    Database(#[from] DatabaseError),
    
    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),
    
    #[error("Queue error: {0}")]
    Queue(#[from] QueueError),
    
    #[error("ML model error: {0}")]
    MLModel(#[from] MLError),
    
    #[error("Integration error: {platform} - {message}")]
    Integration {
        platform: Platform,
        message: String,
        #[source]
        source: Option<Box<dyn StdError + Send + Sync>>,
    },
    
    #[error("Webhook error: {0}")]
    Webhook(String),
    
    #[error("Encryption error: {0}")]
    Encryption(String),
    
    #[error("Parsing error: {0}")]
    Parse(String),
    
    #[error("Conflict: {0}")]
    Conflict(String),
    
    #[error("Unavailable: Service temporarily unavailable")]
    Unavailable,
    
    #[error("Other error: {0}")]
    Other(#[source] Box<dyn StdError + Send + Sync>),
}

impl CoreError {
    /// Get error code for API responses
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Configuration(_) => "CONFIG_ERROR",
            Self::Storage(_) => "STORAGE_ERROR",
            Self::Network(_) => "NETWORK_ERROR",
            Self::Authentication(_) => "AUTH_ERROR",
            Self::Authorization(_) => "AUTHZ_ERROR",
            Self::Validation(_) => "VALIDATION_ERROR",
            Self::NotFound { .. } => "NOT_FOUND",
            Self::AlreadyExists { .. } => "ALREADY_EXISTS",
            Self::RateLimitExceeded(_) => "RATE_LIMITED",
            Self::Timeout(_) => "TIMEOUT",
            Self::Internal(_) => "INTERNAL_ERROR",
            Self::ExternalService { .. } => "EXTERNAL_SERVICE_ERROR",
            Self::Serialization(_) => "SERIALIZATION_ERROR",
            Self::Database(_) => "DATABASE_ERROR",
            Self::Cache(_) => "CACHE_ERROR",
            Self::Queue(_) => "QUEUE_ERROR",
            Self::MLModel(_) => "ML_ERROR",
            Self::Integration { .. } => "INTEGRATION_ERROR",
            Self::Webhook(_) => "WEBHOOK_ERROR",
            Self::Encryption(_) => "ENCRYPTION_ERROR",
            Self::Parse(_) => "PARSE_ERROR",
            Self::Conflict(_) => "CONFLICT",
            Self::Unavailable => "SERVICE_UNAVAILABLE",
            Self::Other(_) => "UNKNOWN_ERROR",
        }
    }
    
    /// Get HTTP status code
    pub fn http_status(&self) -> u16 {
        match self {
            Self::Validation(_) | Self::Parse(_) => 400,
            Self::Authentication(_) => 401,
            Self::Authorization(_) => 403,
            Self::NotFound { .. } => 404,
            Self::AlreadyExists { .. } | Self::Conflict(_) => 409,
            Self::RateLimitExceeded(_) => 429,
            Self::Timeout(_) => 408,
            Self::Unavailable => 503,
            Self::Internal(_) | Self::Other(_) => 500,
            _ => 500,
        }
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network(_) |
            Self::Timeout(_) |
            Self::Unavailable |
            Self::RateLimitExceeded(_) |
            Self::Database(DatabaseError::ConnectionLost) |
            Self::Cache(CacheError::ConnectionFailed) |
            Self::Queue(QueueError::ConnectionLost)
        )
    }
    
    /// Get recovery suggestion for the error
    pub fn recovery_suggestion(&self) -> String {
        match self {
            Self::Configuration(msg) => {
                format!("Check your configuration settings: {}", msg)
            }
            Self::Authentication(_) => {
                "Please check your credentials and try again".to_string()
            }
            Self::Authorization(_) => {
                "You don't have permission to perform this action".to_string()
            }
            Self::RateLimitExceeded(_) => {
                "Too many requests. Please wait and try again later".to_string()
            }
            Self::Timeout(_) => {
                "The operation timed out. Please try again".to_string()
            }
            Self::Network(_) => {
                "Network error occurred. Please check your connection".to_string()
            }
            Self::NotFound { resource_type, .. } => {
                format!("The requested {} was not found", resource_type)
            }
            Self::AlreadyExists { resource_type, .. } => {
                format!("A {} with this ID already exists", resource_type)
            }
            Self::Unavailable => {
                "Service is temporarily unavailable. Please try again later".to_string()
            }
            _ => "An error occurred. Please try again or contact support".to_string(),
        }
    }
}

/// Storage-related errors
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Connection failed")]
    ConnectionFailed,
    
    #[error("Query failed: {0}")]
    QueryFailed(String),
    
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
    
    #[error("Data corruption detected")]
    DataCorruption,
    
    #[error("Storage quota exceeded")]
    QuotaExceeded,
    
    #[error("Backup failed: {0}")]
    BackupFailed(String),
    
    #[error("Migration failed: {0}")]
    MigrationFailed(String),
}

/// Network-related errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection refused")]
    ConnectionRefused,
    
    #[error("DNS resolution failed")]
    DnsResolutionFailed,
    
    #[error("SSL/TLS error")]
    TlsError,
    
    #[error("Invalid response")]
    InvalidResponse,
    
    #[error("Request timeout")]
    RequestTimeout,
    
    #[error("Connection reset")]
    ConnectionReset,
}

/// Authentication and authorization errors
#[derive(Error, Debug)]
pub enum AuthError {
    #[error("Invalid credentials")]
    InvalidCredentials,
    
    #[error("Token expired")]
    TokenExpired,
    
    #[error("Token invalid")]
    TokenInvalid,
    
    #[error("Insufficient permissions")]
    InsufficientPermissions,
    
    #[error("Account locked")]
    AccountLocked,
    
    #[error("Account not verified")]
    AccountNotVerified,
    
    #[error("MFA required")]
    MfaRequired,
    
    #[error("MFA failed")]
    MfaFailed,
    
    #[error("Session expired")]
    SessionExpired,
}

/// Database-specific errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection lost")]
    ConnectionLost,
    
    #[error("Connection pool exhausted")]
    PoolExhausted,
    
    #[error("Deadlock detected")]
    Deadlock,
    
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("Migration pending")]
    MigrationPending,
    
    #[error("Replica lag too high")]
    ReplicaLag,
}

/// Cache-specific errors
#[derive(Error, Debug)]
pub enum CacheError {
    #[error("Connection failed")]
    ConnectionFailed,
    
    #[error("Key not found")]
    KeyNotFound,
    
    #[error("Serialization failed")]
    SerializationFailed,
    
    #[error("TTL expired")]
    TtlExpired,
    
    #[error("Cache full")]
    CacheFull,
    
    #[error("Invalid key format")]
    InvalidKeyFormat,
}

/// Queue-specific errors
#[derive(Error, Debug)]
pub enum QueueError {
    #[error("Connection lost")]
    ConnectionLost,
    
    #[error("Queue full")]
    QueueFull,
    
    #[error("Message too large")]
    MessageTooLarge,
    
    #[error("Consumer timeout")]
    ConsumerTimeout,
    
    #[error("Dead letter queue")]
    DeadLetterQueue,
    
    #[error("Invalid message format")]
    InvalidMessageFormat,
}

/// ML model errors
#[derive(Error, Debug)]
pub enum MLError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),
    
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Invalid input shape")]
    InvalidInputShape,
    
    #[error("Model version mismatch")]
    VersionMismatch,
    
    #[error("Insufficient training data")]
    InsufficientData,
    
    #[error("GPU not available")]
    GpuNotAvailable,
}

/// Error context for additional debugging information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub workspace_id: Option<WorkspaceId>,
    pub user_id: Option<UserId>,
    pub outcome_id: Option<OutcomeId>,
    pub platform: Option<Platform>,
    pub request_id: Option<String>,
    pub trace_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self {
            workspace_id: None,
            user_id: None,
            outcome_id: None,
            platform: None,
            request_id: None,
            trace_id: None,
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }
    
    pub fn with_user(mut self, user_id: UserId) -> Self {
        self.user_id = Some(user_id);
        self
    }
    
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Extended error with context
#[derive(Debug)]
pub struct ContextualError {
    pub error: CoreError,
    pub context: ErrorContext,
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [Context: {:?}]", self.error, self.context)
    }
}

impl StdError for ContextualError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.error.source()
    }
}

/// Result type alias with CoreError
pub type Result<T> = std::result::Result<T, CoreError>;

/// Result type alias with ContextualError
pub type ContextualResult<T> = std::result::Result<T, ContextualError>;

/// Error reporting trait for telemetry
pub trait ErrorReporter: Send + Sync {
    fn report(&self, error: &ContextualError);
    fn report_panic(&self, panic_info: &std::panic::PanicHookInfo);
}

/// Default error reporter implementation
pub struct DefaultErrorReporter;

impl ErrorReporter for DefaultErrorReporter {
    fn report(&self, error: &ContextualError) {
        tracing::error!(
            error_code = error.error.error_code(),
            http_status = error.error.http_status(),
            is_retryable = error.error.is_retryable(),
            context = ?error.context,
            "{}", error.error
        );
    }
    
    fn report_panic(&self, panic_info: &std::panic::PanicHookInfo) {
        tracing::error!(
            panic = true,
            location = ?panic_info.location(),
            "Panic occurred: {:?}", panic_info.payload()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_codes() {
        let err = CoreError::NotFound {
            resource_type: "Outcome".to_string(),
            resource_id: "123".to_string(),
        };
        assert_eq!(err.error_code(), "NOT_FOUND");
        assert_eq!(err.http_status(), 404);
    }
    
    #[test]
    fn test_retryable_errors() {
        assert!(CoreError::Network(NetworkError::ConnectionRefused).is_retryable());
        assert!(CoreError::RateLimitExceeded("test".to_string()).is_retryable());
        assert!(!CoreError::Validation("test".to_string()).is_retryable());
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new()
            .with_workspace(WorkspaceId::new())
            .with_user(UserId::from("user123"))
            .with_request_id("req-123".to_string());
        
        assert!(context.workspace_id.is_some());
        assert!(context.user_id.is_some());
        assert_eq!(context.request_id, Some("req-123".to_string()));
    }
}