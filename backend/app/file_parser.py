"""
File parsing utilities for extracting text from various file formats.
"""
import io
from pathlib import Path
from typing import Optional

from PyPDF2 import PdfReader
from docx import Document

from .exceptions import FileProcessingError, UnsupportedFileTypeError, FileSizeError


ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file(filename: str, content: bytes) -> None:
    """
    Validate file type and size.
    
    Args:
        filename: Name of the file
        content: File content as bytes
        
    Raises:
        UnsupportedFileTypeError: If file type is not supported
        FileSizeError: If file size exceeds limit
    """
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(filename, list(ALLOWED_EXTENSIONS))
    
    file_size_bytes = len(content)
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        raise FileSizeError(filename, size_mb, MAX_FILE_SIZE_MB)


def extract_text_from_txt(content: bytes, filename: str) -> str:
    """
    Extract text from TXT file.
    
    Args:
        content: File content as bytes
        filename: Name of the file (for error messages)
        
    Returns:
        Extracted text string
        
    Raises:
        FileProcessingError: If text extraction fails
    """
    try:
        text = content.decode('utf-8')
        return text.strip()
    except UnicodeDecodeError:
        try:
            text = content.decode('latin-1')
            return text.strip()
        except Exception as e:
            raise FileProcessingError(f"Unable to decode text file: {e}", filename)
    except Exception as e:
        raise FileProcessingError(f"Failed to read TXT file: {e}", filename)


def extract_text_from_pdf(content: bytes, filename: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        content: File content as bytes
        filename: Name of the file (for error messages)
        
    Returns:
        Extracted text string
        
    Raises:
        FileProcessingError: If text extraction fails
    """
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        
        if len(pdf_reader.pages) == 0:
            raise FileProcessingError("PDF file contains no pages", filename)
        
        text_parts = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        text = "\n".join(text_parts).strip()
        
        if not text:
            raise FileProcessingError("No text content found in PDF", filename)
        
        return text
        
    except FileProcessingError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Failed to read PDF file: {e}", filename)


def extract_text_from_docx(content: bytes, filename: str) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        content: File content as bytes
        filename: Name of the file (for error messages)
        
    Returns:
        Extracted text string
        
    Raises:
        FileProcessingError: If text extraction fails
    """
    try:
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text_parts.append(paragraph.text)
        
        text = "\n".join(text_parts).strip()
        
        if not text:
            raise FileProcessingError("No text content found in DOCX", filename)
        
        return text
        
    except FileProcessingError:
        raise
    except Exception as e:
        raise FileProcessingError(f"Failed to read DOCX file: {e}", filename)


def extract_text_from_file(content: bytes, filename: str) -> str:
    """
    Extract text from file based on extension.
    
    Args:
        content: File content as bytes
        filename: Name of the file
        
    Returns:
        Extracted text string
        
    Raises:
        UnsupportedFileTypeError: If file type is not supported
        FileSizeError: If file size exceeds limit
        FileProcessingError: If text extraction fails
    """
    validate_file(filename, content)
    
    file_ext = Path(filename).suffix.lower()
    
    if file_ext == ".txt":
        return extract_text_from_txt(content, filename)
    elif file_ext == ".pdf":
        return extract_text_from_pdf(content, filename)
    elif file_ext == ".docx":
        return extract_text_from_docx(content, filename)
    else:
        raise UnsupportedFileTypeError(filename, list(ALLOWED_EXTENSIONS))
