from sqlalchemy import Column, Integer, String, Date, Time
from sqlalchemy.orm import declarative_base
from datetime import datetime


# Define the SQLAlchemy ORM base and log model
Base = declarative_base()


class LogEntry(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True)
    date = Column(Date, default=datetime.utcnow().date)
    time = Column(Time, default=datetime.utcnow().time)
    level = Column(String)
    message = Column(String)
    stack_trace = Column(
        String, nullable=True
    )  # Stack trace is nullable and only filled in case of errors
