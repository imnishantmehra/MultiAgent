# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.inspection import inspect
from models import Base, Content, ContentStatus, PlatformEnum
from datetime import datetime
import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

class DatabaseManager:
    def __init__(self):
        # Get database URL from environment variable
        database_url = os.getenv('DATABASE_URL')
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Ensure tables exist before performing operations
        self.create_tables_if_not_exist()

    def create_tables_if_not_exist(self):
        """Check if tables exist and create them if they don't."""
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        
        if 'content' not in table_names:  # Ensure table is present
            print("Table 'content' not found. Creating now...")
            Base.metadata.create_all(self.engine)  # Create missing tables
            print("Table 'content' created successfully.")

    def get_db_session(self):
        """Get a database session"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def store_content(self, content_data: Dict, file_name: str, file_type: str) -> List[Content]:
        """Store generated content in the database, ensuring the table exists."""
        session = next(self.get_db_session())  # Open a session
        stored_contents = []

        try:
            print(f"Available platform enums: {[e.name for e in PlatformEnum]}")

            for platform, posts in content_data.items():
                print(f"Attempting to store platform: {platform}")  # ✅ Moved inside loop

                for post in posts:
                    # Extract week and day from week_day string (e.g., "Week 1 - Monday")
                    week_day_parts = post['week_day'].split(' - ')
                    week = int(week_day_parts[0].split(' ')[1])
                    day = week_day_parts[1]

                    content_entry = Content(
                        week=week,
                        day=day,
                        content=post['content'],
                        title=post['title'],
                        status=ContentStatus.pending,
                        date_upload=datetime.now().date(),
                        platform=PlatformEnum[platform.upper()],
                        file_name=file_name,
                        file_type=file_type
                    )
                    session.add(content_entry)
                    stored_contents.append(content_entry)

            session.commit()
            return stored_contents

        except SQLAlchemyError as e:
            session.rollback()
            raise Exception(f"Error storing content in database: {str(e)}")
        finally:
            session.close()  # Ensure session is closed properly

    def update_content_status(self, content_id: int, status: ContentStatus):
        """Update the status of a content entry"""
        session = next(self.get_db_session())
        try:
            content = session.query(Content).filter(Content.id == content_id).first()
            if content:
                content.status = status
                session.commit()
                return True
            return False
        except SQLAlchemyError as e:
            session.rollback()
            raise Exception(f"Error updating content status: {str(e)}")
        finally:
            session.close()

    def get_pending_content(self) -> List[Content]:
        """Get all pending content"""
        session = next(self.get_db_session())
        try:
            return session.query(Content).filter(Content.status == ContentStatus.pending).all()
        except SQLAlchemyError as e:
            raise Exception(f"Error fetching pending content: {str(e)}")
        finally:
            session.close()

    def get_content_by_platform(self, platform: PlatformEnum) -> List[Content]:
        """Get all content for a specific platform"""
        session = next(self.get_db_session())
        try:
            return session.query(Content).filter(Content.platform == platform).all()
        except SQLAlchemyError as e:
            raise Exception(f"Error fetching content for platform {platform}: {str(e)}")
        finally:
            session.close()

    def get_pending_content_file(self, file_name: str = None):
        """Fetch content from the database, optionally filtered by file_name."""
        session = next(self.get_db_session())  # Get a database session
        
        try:
            query = session.query(Content).filter(Content.status == ContentStatus.pending)
            
            # If a file_name is provided, filter the query by file_name
            if file_name:
                query = query.filter(Content.file_name == file_name)
                
            return query.all()
        except SQLAlchemyError as e:
            raise Exception(f"Error fetching pending content: {str(e)}")
        finally:
            session.close()
