from sqlalchemy import create_engine, Column,  Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker,declarative_base
from datetime import datetime,timezone

engine = create_engine('sqlite:///predictions.db')

Base = declarative_base()

# This defines table - each attribute = each column

'''
create_engine    →  connects Python to the database file
declarative_base →  creates a registry for all your tables
Column           →  defines one column in a table
class Prediction →  defines the entire table structure
create_all       →  physically creates the tables in the file
sessionmaker     →  factory for creating sessions
SessionLocal()   →  one conversation with the database
'''

class Prediction(Base):
    # To find the table name
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime,default=lambda: datetime.now(timezone.utc))

# This actually creates the table in the database file
Base.metadata.create_all(bind=engine)

# This is how your API gets a database connection
SessionLocal = sessionmaker(bind=engine)

