use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Extraction failed: {0}")]
    ExtractionError(String),

    #[error("Outcome mapping failed: {0}")]
    MappingError(String),

    #[error("Platform {0} not supported")]
    UnsupportedPlatform(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
