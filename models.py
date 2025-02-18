# models.py
from sqlalchemy import Column, Integer, String, Date, SmallInteger, Enum
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class PlatformEnum(enum.Enum):
    LINKEDIN = "LINKEDIN"
    INSTAGRAM = "INSTAGRAM"
    FACEBOOK = "FACEBOOK"
    TWITTER = "TWITTER"
    WORDPRESS = "WORDPRESS"
    YOUTUBE =   "YOUTUBE"
    TIKTOK = "TIKTOK"

class ContentStatus(enum.Enum):
    pending= "pending"
    uploaded = "uploaded"

class Content(Base):
    __tablename__ = 'content'

    id = Column(SmallInteger, primary_key=True, autoincrement=True)
    week = Column(Integer, nullable=False)
    day = Column(String(20), nullable=False)
    content = Column(String, nullable=False)
    title = Column(String(255), nullable=False)
    status = Column(Enum(ContentStatus), default=ContentStatus.pending)
    date_upload = Column(Date, nullable=False)
    platform = Column(Enum(PlatformEnum), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)

    def __repr__(self):
        return f"<Content(id={self.id}, week={self.week}, day={self.day}, platform={self.platform})>"