import logging
import os
from logging.handlers import BufferingHandler
from sqlalchemy import create_engine, Column, Integer, String, Sequence, Text, Date, Time
from sqlalchemy.orm import sessionmaker, declarative_base
import traceback
from datetime import datetime

Base = declarative_base()


class LogEntry(Base):
    __tablename__ = "logs"
    id = Column(Integer, Sequence("log_id_seq"), primary_key=True)
    date = Column(Date)
    time = Column(Time)
    level = Column(String(50))
    message = Column(String(255))
    stack_trace = Column(Text, nullable=True)


class SQLAlchemyHandler(BufferingHandler):
    def __init__(self, engine, capacity=1):  # Set capacity to 1 to immediately flush each log
        super().__init__(capacity)
        self.engine = engine
        self.Session = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)

    def flush(self):
        session = self.Session()
        for record in self.buffer:
            stack_trace = None
            if record.exc_info:
                stack_trace = "".join(traceback.format_exception(*record.exc_info))
            log_datetime = datetime.fromtimestamp(record.created)
            log_entry = LogEntry(
                date=log_datetime.date(),
                time=log_datetime.time(),
                level=record.levelname,
                message=record.getMessage(),
                stack_trace=stack_trace,
            )
            session.add(log_entry)
        session.commit()
        self.buffer = []


# Setup the database engine with a busy timeout of 5 seconds
base_dir = os.path.dirname(os.path.abspath(__file__))
engine = create_engine(
    f'sqlite:///{os.path.join(base_dir, "dicom_viewer_logs.db")}', connect_args={"timeout": 5}
)

# Setup logging
logger = logging.getLogger("DICOMViewer")
logger.setLevel(logging.DEBUG)
db_handler = SQLAlchemyHandler(engine)
logger.addHandler(db_handler)
