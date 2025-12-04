import logging
import logging.handlers
from datetime import datetime, timedelta
import os
import json
import io
import sys

class ModelLogger:
    def __init__(self, log_dir='.'):
        self.log_dir = log_dir
        self.setup_loggers()

    def setup_loggers(self):
        """Set up the three required log files with rotation and cleanup."""
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Model actions logger
        self.model_actions_logger = self._create_logger(
            'model_actions',
            os.path.join(self.log_dir, 'model_actions.log'),
            'Model Actions'
        )

        # User feedback logger
        self.user_feedback_logger = self._create_logger(
            'user_feedback',
            os.path.join(self.log_dir, 'user_feedback.log'),
            'User Feedback'
        )

        # Error reports logger
        self.error_logger = self._create_logger(
            'error_reports',
            os.path.join(self.log_dir, 'error_reports.log'),
            'Error Reports'
        )

    def _create_logger(self, name, log_file, logger_name):
        """Create a logger with timed rotation and cleanup."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create timed rotating file handler (rotates every 30 days)
        # Note: Python's TimedRotatingFileHandler rotates based on time,
        # but we need to implement custom cleanup for exactly 30 days
        handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=30,  # Every 30 days
            backupCount=0  # We'll handle cleanup manually
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def log_model_action(self, action, details=None):
        """Log model-related actions."""
        message = f"ACTION: {action}"
        if details:
            message += f" - DETAILS: {details}"
        self.model_actions_logger.info(message)

    def log_user_feedback(self, message, predicted, correct, confidence):
        """Log user feedback."""
        feedback_data = {
            "message": message,
            "predicted": predicted,
            "correct": correct,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.user_feedback_logger.info(f"FEEDBACK: {json.dumps(feedback_data)}")

    def log_error(self, error_type, message, details=None):
        """Log errors and issues."""
        error_message = f"ERROR_TYPE: {error_type} - MESSAGE: {message}"
        if details:
            error_message += f" - DETAILS: {details}"
        self.error_logger.error(error_message)

    def cleanup_old_logs(self, days=30):
        """Clean up log entries older than specified days from all log files."""
        cutoff_date = datetime.now() - timedelta(days=days)

        log_files = [
            'model_actions.log',
            'user_feedback.log',
            'error_reports.log'
        ]

        total_cleaned = 0

        for log_file in log_files:
            log_path = os.path.join(self.log_dir, log_file)
            if not os.path.exists(log_path):
                continue

            cleaned_lines = []
            cleaned_count = 0

            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # Extract timestamp from log line
                        # Format: "2023-10-19 10:30:45,123 - LoggerName - Level - Message"
                        try:
                            timestamp_str = line.split(' - ')[0]
                            log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')

                            if log_date > cutoff_date:
                                cleaned_lines.append(line)
                            else:
                                cleaned_count += 1
                        except (ValueError, IndexError):
                            # If we can't parse the timestamp, keep the line
                            cleaned_lines.append(line)

                # Rewrite the file with only recent entries
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.writelines(cleaned_lines)

                if cleaned_count > 0:
                    print(f"Cleaned {cleaned_count} old entries from {log_file}")

                total_cleaned += cleaned_count

            except Exception as e:
                self.log_error("LOG_CLEANUP_ERROR", f"Failed to cleanup {log_file}", str(e))

        # Also cleanup echo logs
        echo_cleaned = echo_logger.cleanup_old_echo_logs(days)
        if echo_cleaned > 0:
            print(f"Cleaned {echo_cleaned} old entries from echo_interactions.log")
            total_cleaned += echo_cleaned

        if total_cleaned > 0:
            print(f"Total cleaned entries across all logs: {total_cleaned}")
        else:
            print("No old log entries to clean up.")

    def get_recent_logs(self, log_type, count=5):
        """Get the last N entries from a specific log file."""
        log_files = {
            'model_actions': 'model_actions.log',
            'user_feedback': 'user_feedback.log',
            'error_reports': 'error_reports.log'
        }

        if log_type not in log_files:
            return ["Invalid log type. Use: model_actions, user_feedback, or error_reports"]

        log_path = os.path.join(self.log_dir, log_files[log_type])

        if not os.path.exists(log_path):
            return ["Log file does not exist"]

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Get last N lines
            recent_lines = lines[-count:] if len(lines) >= count else lines

            return [line.strip() for line in recent_lines]

        except Exception as e:
            return [f"Error reading log file: {str(e)}"]

    def get_log_stats(self):
        """Get statistics about all log files."""
        stats = {}

        log_files = {
            'model_actions': 'model_actions.log',
            'user_feedback': 'user_feedback.log',
            'error_reports': 'error_reports.log'
        }

        for log_type, filename in log_files.items():
            log_path = os.path.join(self.log_dir, filename)

            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    stats[log_type] = {
                        'total_entries': len(lines),
                        'file_size_kb': os.path.getsize(log_path) / 1024,
                        'last_modified': datetime.fromtimestamp(
                            os.path.getmtime(log_path)
                        ).isoformat()
                    }
                except Exception as e:
                    stats[log_type] = {'error': str(e)}
            else:
                stats[log_type] = {'total_entries': 0, 'file_size_kb': 0}

        return stats

class EchoLogger:
    """Handles user echo logging functionality."""

    def __init__(self, log_dir='.'):
        self.log_dir = log_dir
        self.echo_enabled = False
        self.current_session = 0
        self.session_file = os.path.join(log_dir, 'echo_session.txt')
        self.state_file = os.path.join(log_dir, 'echo_state.txt')
        self.setup_echo_logger()
        self.load_session_state()

    def setup_echo_logger(self):
        """Set up the echo interactions logger with rotation and cleanup."""
        os.makedirs(self.log_dir, exist_ok=True)

        self.echo_logger = logging.getLogger('echo_interactions')
        self.echo_logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(message)s')  # Custom format for echo logs

        # Create timed rotating file handler
        handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(self.log_dir, 'echo_interactions.log'),
            when='midnight',
            interval=30,
            backupCount=0
        )
        handler.setFormatter(formatter)
        self.echo_logger.addHandler(handler)

    def load_session_state(self):
        """Load the current session number and echo state."""
        try:
            # Load session number
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    self.current_session = int(f.read().strip())
            else:
                self.current_session = 0

            # Load echo state
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = f.read().strip().lower()
                    self.echo_enabled = state == 'enabled'
            else:
                self.echo_enabled = False
        except Exception as e:
            logger.log_error("ECHO_LOG_LOAD_ERROR", f"Failed to load echo log state: {str(e)}")

    def save_session_state(self):
        """Save the current session number and echo state."""
        try:
            with open(self.session_file, 'w') as f:
                f.write(str(self.current_session))
            with open(self.state_file, 'w') as f:
                f.write('enabled' if self.echo_enabled else 'disabled')
        except Exception as e:
            logger.log_error("ECHO_LOG_SAVE_ERROR", f"Failed to save echo log state: {str(e)}")

    def start_session(self):
        """Start a new echo logging session."""
        if not self.echo_enabled:
            return

        self.current_session += 1
        timestamp = datetime.now().isoformat()

        # Log session start
        self.echo_logger.info(f"SESSION: {self.current_session} - ECHO_LOG_ENABLED - TIMESTAMP: {timestamp}")

        # Log to model actions
        logger.log_model_action("ECHO_LOG_ENABLED", f"Session {self.current_session} started at {timestamp}")

        self.save_session_state()

    def end_session(self):
        """End the current echo logging session."""
        if not self.echo_enabled:
            return

        timestamp = datetime.now().isoformat()

        # Log session end
        self.echo_logger.info(f"SESSION: {self.current_session} - ECHO_LOG_DISABLED - TIMESTAMP: {timestamp}")

        # Log to model actions
        logger.log_model_action("ECHO_LOG_DISABLED", f"Session {self.current_session} ended at {timestamp}")

        self.echo_enabled = False
        self.save_session_state()

    def enable_echo_log(self):
        """Enable echo logging."""
        if self.echo_enabled:
            return False  # Already enabled

        self.echo_enabled = True
        self.start_session()
        return True

    def disable_echo_log(self):
        """Disable echo logging."""
        if not self.echo_enabled:
            return False  # Already disabled

        self.end_session()
        return True

    def log_interaction(self, interaction_type, content):
        """Log a user interaction or system output."""
        if not self.echo_enabled:
            return

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"SESSION: {self.current_session} - TIMESTAMP: {timestamp} - TYPE: {interaction_type} - CONTENT: {content}"
        self.echo_logger.info(log_entry)

    def cleanup_old_echo_logs(self, days=30):
        """Clean up echo log entries older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        log_path = os.path.join(self.log_dir, 'echo_interactions.log')

        if not os.path.exists(log_path):
            return 0

        cleaned_lines = []
        cleaned_count = 0

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Extract timestamp from echo log line
                    # Format: "SESSION: X - TIMESTAMP: YYYY-MM-DD HH:MM:SS - TYPE: ... - CONTENT: ..."
                    try:
                        if 'TIMESTAMP:' in line:
                            timestamp_str = line.split('TIMESTAMP:')[1].split(' - ')[0].strip()
                            log_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                            if log_date > cutoff_date:
                                cleaned_lines.append(line)
                            else:
                                cleaned_count += 1
                        else:
                            # Keep lines without timestamps (like session markers)
                            cleaned_lines.append(line)
                    except (ValueError, IndexError):
                        cleaned_lines.append(line)

            # Rewrite the file with only recent entries
            with open(log_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)

        except Exception as e:
            logger.log_error("ECHO_LOG_CLEANUP_ERROR", f"Failed to cleanup echo logs: {str(e)}")

        return cleaned_count

# Global logger instances
logger = ModelLogger()
echo_logger = EchoLogger()

def test_logging_system():
    """Test the logging system functionality."""
    print("Testing Logging System...")

    # Log some test entries
    logger.log_model_action("MODEL_TRAINED", "SVM model trained with 99.3% accuracy")
    logger.log_model_action("MODEL_LOADED", "SVM_20251018_214404.pkl loaded successfully")
    logger.log_user_feedback("Test spam message", "spam", "spam", 0.95)
    logger.log_error("TEST_ERROR", "This is a test error", "Additional details")

    # Get recent logs
    print("\nLast 3 model actions:")
    for log in logger.get_recent_logs('model_actions', 3):
        print(f"  {log}")

    print("\nLast 3 user feedback:")
    for log in logger.get_recent_logs('user_feedback', 3):
        print(f"  {log}")

    print("\nLast 3 error reports:")
    for log in logger.get_recent_logs('error_reports', 3):
        print(f"  {log}")

    # Get stats
    print("\nLog Statistics:")
    stats = logger.get_log_stats()
    for log_type, stat in stats.items():
        if 'error' not in stat:
            print(f"  {log_type}: {stat['total_entries']} entries, {stat['file_size_kb']:.1f} KB")
        else:
            print(f"  {log_type}: Error - {stat['error']}")

    # Test cleanup (won't remove anything since entries are new)
    print("\nTesting cleanup (should find no old entries):")
    logger.cleanup_old_logs()

if __name__ == "__main__":
    test_logging_system()