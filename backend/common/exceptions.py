"""
Custom exceptions used throughout the Drishti project.
"""


class DrishtiError(Exception):
    """Base exception for all project-specific errors."""

    def __init__(self, message: str = "An unexpected Drishti error occurred"):
        super().__init__(message)


class ConfigurationError(DrishtiError):
    """Raised when configuration is invalid or missing."""


class ValidationError(DrishtiError):
    """Raised when dataset validation fails."""


class DatasetNotFoundError(DrishtiError):
    """Raised when a required dataset cannot be located."""


class UnsupportedFileTypeError(DrishtiError):
    """Raised when an unsupported file format is encountered."""


class EmptyDatasetError(DrishtiError):
    """Raised when a dataset contains no usable records."""


class MissingColumnError(ValidationError):
    """Raised when required columns are missing."""

    def __init__(self, columns: list[str]):
        message = f"Missing required columns: {', '.join(columns)}"
        super().__init__(message)
        self.columns = columns


class DuplicateRecordError(ValidationError):
    """Raised when duplicate records are detected."""


class DataIntegrityError(ValidationError):
    """Raised when dataset integrity checks fail."""


class ProcessingError(DrishtiError):
    """Raised during ETL processing."""


class FeatureEngineeringError(ProcessingError):
    """Raised during feature engineering."""


class DatabaseConnectionError(DrishtiError):
    """Raised when a database connection cannot be established."""


class DatabaseWriteError(DrishtiError):
    """Raised when writing data to the database fails."""


class APIError(DrishtiError):
    """Raised for API-related failures."""


class ModelTrainingError(DrishtiError):
    """Raised when machine learning model training fails."""


class PredictionError(DrishtiError):
    """Raised when inference or prediction fails."""


class FileWriteError(DrishtiError):
    """Raised when a file cannot be written."""


class FileReadError(DrishtiError):
    """Raised when a file cannot be read."""


class DirectoryCreationError(DrishtiError):
    """Raised when directory creation fails."""


class InvalidConfigurationValue(ConfigurationError):
    """Raised when a configuration value is invalid."""


class MissingEnvironmentVariable(ConfigurationError):
    """Raised when a required environment variable is missing."""

    def __init__(self, variable: str):
        message = f"Missing required environment variable: {variable}"
        super().__init__(message)
        self.variable = variable