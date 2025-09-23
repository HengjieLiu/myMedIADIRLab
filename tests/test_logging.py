import pytest
from dicom_viewer.config.logging_config import logger, SQLAlchemyHandler, engine
from dicom_viewer.models.log_entry import LogEntry
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def log_handler():
    """Fixture to access SQLAlchemy logging handler."""
    return next(handler for handler in logger.handlers if isinstance(handler, SQLAlchemyHandler))


def test_logger_is_setup_correctly(log_handler):
    """Test if the logger is setup correctly with SQLAlchemyHandler."""
    assert isinstance(log_handler, SQLAlchemyHandler)


def test_logging_to_database(log_handler):
    """Test logging messages to the database."""
    # Prepare the database session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Log a test message
    logger.info("Test log message")

    # Flush the log handler to ensure the message is written to the database
    log_handler.flush()

    # Check if the message was logged in the database
    log_entry = session.query(LogEntry).filter_by(message="Test log message").first()
    assert log_entry is not None
    assert log_entry.message == "Test log message"
    assert log_entry.level == "INFO"

    # Clean up by deleting the test log entry
    session.delete(log_entry)
    session.commit()
    session.close()
