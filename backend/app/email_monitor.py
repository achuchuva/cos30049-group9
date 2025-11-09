"""
Email monitoring module for Outlook IMAP spam detection.
Simple, isolated implementation that polls for new emails and detects spam.
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque

from imap_tools import MailBox, AND
from .config import settings
from . import database

logger = logging.getLogger(__name__)


class EmailMonitor:
    """Monitors Outlook email via IMAP and detects spam in incoming messages."""
    
    def __init__(self, predictor):
        """
        Initialize the email monitor.
        
        Args:
            predictor: SpamPredictor instance for spam detection
        """
        self.predictor = predictor
        self.host = settings.EMAIL_IMAP_HOST
        self.port = settings.EMAIL_IMAP_PORT
        self.email = settings.EMAIL_ADDRESS
        self.password = settings.EMAIL_PASSWORD
        self.poll_interval = settings.EMAIL_POLL_INTERVAL
        self.enabled = settings.EMAIL_MONITORING_ENABLED
        
        # In-memory storage for recent detections (last 50)
        self.recent_detections: deque = deque(maxlen=50)
        self.last_check_time: Optional[datetime] = None
        self.is_running = False
        self.monitor_task = None
        
        logger.info(f"Email monitor initialized. Enabled: {self.enabled}")
    
    async def start(self):
        """Start the email monitoring task."""
        if not self.enabled:
            logger.info("Email monitoring is disabled in settings")
            return
        
        if not self.email or not self.password:
            logger.warning("Email credentials not configured. Monitoring disabled.")
            return
        
        if self.is_running:
            logger.warning("Email monitor is already running")
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Email monitoring started. Polling every {self.poll_interval} seconds")
    
    async def stop(self):
        """Stop the email monitoring task."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Email monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop that polls for new emails."""
        while self.is_running:
            try:
                await self._check_new_emails()
            except Exception as e:
                logger.error(f"Error in email monitoring loop: {e}", exc_info=True)
            
            # Wait for next poll interval
            await asyncio.sleep(self.poll_interval)
    
    async def _check_new_emails(self):
        """Check for new emails and process them."""
        try:
            # Run IMAP operations in thread pool to avoid blocking
            await asyncio.to_thread(self._fetch_and_process_emails)
        except Exception as e:
            logger.error(f"Error checking emails: {e}", exc_info=True)
    
    def _fetch_and_process_emails(self):
        """Fetch and process new emails from IMAP server."""
        try:
            with MailBox(self.host).login(self.email, self.password) as mailbox:
                # Get current time for filtering
                current_time = datetime.utcnow()
                
                # If first run, set last_check_time to now and don't process old emails
                if self.last_check_time is None:
                    self.last_check_time = current_time
                    logger.info("First email check - setting baseline time")
                    return
                
                # Fetch unseen emails since the last check
                criteria = AND(date_gte=self.last_check_time.date())
                messages = list(mailbox.fetch(criteria, mark_seen=False, limit=25))
                
                if messages:
                    logger.info(f"Found {len(messages)} unseen email(s)")
                
                for msg in messages:
                    self._process_email(msg)
                
                # Update last check time
                self.last_check_time = current_time
                
        except Exception as e:
            logger.error(f"IMAP connection error: {e}", exc_info=True)
    
    def _process_email(self, msg):
        """
        Process a single email message and detect spam.
        
        Args:
            msg: Email message from imap_tools
        """
        try:
            # Use email UID to prevent duplicate processing
            email_uid = msg.uid
            if database.is_email_processed(email_uid):
                logger.info(f"Skipping already processed email UID: {email_uid}")
                return

            # Extract email content
            subject = msg.subject or ""
            sender = msg.from_ or ""
            body = msg.text or msg.html or ""
            
            # Combine subject and body for spam detection
            email_text = f"Subject: {subject}\n\n{body}"
            
            # Truncate if too long
            if len(email_text) > 10000:
                email_text = email_text[:10000]
            
            # Run spam detection
            result = self.predictor.predict(email_text)
            features = self.predictor.extract_numerical_features(email_text)
            
            # Create detection record
            timestamp = datetime.utcnow().isoformat()
            text_preview = email_text[:200] + "..." if len(email_text) > 200 else email_text
            
            detection = {
                "timestamp": timestamp,
                "subject": subject[:200],  # Truncate long subjects
                "sender": sender,
                "is_spam": result["is_spam"],
                "prediction": result["label"],
                "confidence": result["confidence"],
                "spam_probability": result["spam_probability"],
                "ham_probability": result["ham_probability"],
                "features": features,
                "text_preview": text_preview
            }

            # Save to database
            database.save_prediction(
                text_preview=text_preview,
                prediction=result["label"],
                confidence=result["confidence"],
                is_spam=result["is_spam"],
                spam_probability=result["spam_probability"],
                ham_probability=result["ham_probability"],
                features=features,
                source_type="email",
                timestamp=timestamp
            )

            # Mark email as processed
            database.add_processed_email(email_uid)
            
            # Add to recent in-memory detections
            self.recent_detections.append(detection)
            
            # Log result
            status = "SPAM" if result["is_spam"] else "HAM"
            logger.info(
                f"Email processed and saved - From: {sender}, Subject: {subject[:50]}, "
                f"Result: {status}, Confidence: {result['confidence']:.2%}"
            )
            
        except Exception as e:
            logger.error(f"Error processing email: {e}", exc_info=True)
    
    def get_recent_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent email spam detections.
        
        Args:
            limit: Maximum number of detections to return
            
        Returns:
            List of detection records
        """
        detections = list(self.recent_detections)
        detections.reverse()  # Most recent first
        return detections[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get email monitoring statistics.
        
        Returns:
            Dictionary with monitoring stats
        """
        detections = list(self.recent_detections)
        spam_count = sum(1 for d in detections if d["is_spam"])
        
        return {
            "enabled": self.enabled,
            "is_running": self.is_running,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "total_processed": len(detections),
            "spam_detected": spam_count,
            "ham_detected": len(detections) - spam_count,
            "poll_interval": self.poll_interval
        }
