"""
Custom exception classes for API error handling.
"""
from fastapi import HTTPException, status


class FileProcessingError(HTTPException):
    """Raised when file processing fails."""
    def __init__(self, detail: str, file_name: str = None):
        msg = f"File processing failed: {detail}"
        if file_name:
            msg = f"Failed to process '{file_name}': {detail}"
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=msg
        )


class UnsupportedFileTypeError(HTTPException):
    """Raised when file type is not supported."""
    def __init__(self, file_name: str, allowed_types: list[str]):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File '{file_name}' has unsupported type. Allowed types: {', '.join(allowed_types)}"
        )


class FileSizeError(HTTPException):
    """Raised when file size exceeds limit."""
    def __init__(self, file_name: str, size_mb: float, max_mb: int = 5):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{file_name}' ({size_mb:.2f}MB) exceeds maximum size of {max_mb}MB"
        )


class ModelNotLoadedError(HTTPException):
    """Raised when model is not loaded."""
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact the administrator."
        )


class EmptyTextError(HTTPException):
    """Raised when text input is empty."""
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty"
        )


class TextTooLongError(HTTPException):
    """Raised when text exceeds maximum length."""
    def __init__(self, length: int, max_length: int = 10000):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text is too long ({length} characters). Maximum length is {max_length} characters."
        )
