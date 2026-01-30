use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChronosError {
    #[error("insufficient data: {0}")]
    InsufficientData(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("model error: {0}")]
    ModelError(String),

    #[error("configuration error: {0}")]
    ConfigError(String),

    #[error("analysis error: {0}")]
    AnalysisError(String),

    #[error("normalization error: {0}")]
    NormalizationError(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ChronosError>;
