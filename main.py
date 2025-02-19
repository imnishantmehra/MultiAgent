import os
from fastapi import FastAPI, UploadFile, File, HTTPException , Query, Form
import json
from typing import Optional, Dict, List, Union, Any
from pydantic import BaseModel
from agents import (
    script_research_agent, qc_agent, script_rewriter_agent, regenrate_content_agent, regenrate_subcontent_agent,
    instagram_agent, linkedin_agent, facebook_agent, twitter_agent, wordpress_agent, youtube_agent, tiktok_agent,
    PLATFORM_LIMITS
)
from tasks import (
    script_research_task, qc_task, script_rewriter_task, regenrate_content_task, regenrate_subcontent_task,
    linkedin_task, instagram_task, facebook_task, twitter_task, wordpress_task, youtube_task, tiktok_task
)
from crewai import Crew, Process
from tools import process_content_for_platform, extract_title_from_content, generate_unique_content,generate_different_content, FileProcessor
from database import DatabaseManager
from pathlib import Path
from models import Content, ContentStatus, PlatformEnum
from fastapi.middleware.cors import CORSMiddleware
import random
from threading import Timer
import time
from datetime import datetime, timedelta
from io import BytesIO
import re


app = FastAPI()
# Initialize database manager
db_manager = DatabaseManager()


# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Adjust this to specify allowed HTTP methods
    allow_headers=["*"],  # Adjust this to specify allowed headers
)

# Ensure the uploads directory exists
UPLOAD_DIR = './uploads'
OUTPUT_DIR = './outputs'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ContentResponse(BaseModel):
    week_day: str
    title: str
    content: str
    platform: str
    timestamp: str
    word_count: int
    char_count: int

def save_output_to_file(data: Dict, filename: str) -> str:
    """Save generated content to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{filename}_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return output_path

# # Word/Character count limits for each platform
# PLATFORM_LIMITS = {
#     "twitter": {"chars": 280, "words": None},
#     "instagram": {"chars": None, "words": 400},
#     "linkedin": {"chars": None, "words": 600},
#     "facebook": {"chars": None, "words": 1000},
#     "wordpress": {"chars": None, "words": 2000}
# }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server and save it in the uploads folder.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename) 
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        print(f"File successfully uploaded: {file_path}")
        return {"file_path": file_path, "message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

@app.post("/generate_social_media_scripts")
async def generate_social_media_scripts(
    file: UploadFile = File(...),
    weeks: int = 1,
    platform: str = "all"
) -> Dict:
    try:
        # Save and extract text from file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get file type
        file_type = Path(file.filename).suffix.lstrip('.')

        processor = FileProcessor()
        try:
            extracted_text = processor.extract_text_from_file(file_path)
            print(f"Successfully extracted text from {file.filename}")
            print(f"Extracted text length: {len(extracted_text)} characters")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file: {str(e)}"
            )

        # Research Phase
        research_crew = Crew(
            agents=[script_research_agent],
            tasks=[script_research_task],
            process=Process.sequential
        )
        research_result = research_crew.kickoff(
            inputs={
                "text": extracted_text,
                "file_path": file_path
            }
        )
        researched_content = research_result['output'] if isinstance(research_result, dict) else extracted_text

        # QC Phase
        qc_crew = Crew(
            agents=[qc_agent],
            tasks=[qc_task],
            process=Process.sequential
        )
        qc_result = qc_crew.kickoff(
            inputs={
                "text": researched_content
            }
        )
        cleaned_content = qc_result['output'] if isinstance(qc_result, dict) else researched_content

        # Platform selection
        platforms = {
            "linkedin": (linkedin_agent, linkedin_task),
            "instagram": (instagram_agent, instagram_task),
            "facebook": (facebook_agent, facebook_task),
            "twitter": (twitter_agent, twitter_task),
            "wordpress": (wordpress_agent, wordpress_task),
            "youtube": (youtube_agent, youtube_task),
            "tiktok": (tiktok_agent, tiktok_task)
        }

        if platform != "all" and platform not in platforms:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid platform: {platform}. Available platforms: {', '.join(platforms.keys())}"
            )

        selected_platforms = platforms.items() if platform == "all" else [(platform, platforms[platform])]
        
        # Generate content for each platform
        results = {}
        for platform_name, (agent, task) in selected_platforms:
            platform_crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential
            )
            
            platform_posts = []
            for week in range(1, weeks + 1):
                for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "saturday", "sunday"]:
                    # Generate unique content for each day
                    crew_result = platform_crew.kickoff(
                        inputs={
                            "text": cleaned_content,
                            "day": day,
                            "week": week,
                            "platform": platform_name,
                            "limits": PLATFORM_LIMITS[platform_name]
                        }
                    )
                    
                    # Generate unique content based on the day and week
                    raw_content = generate_unique_content(
                        crew_result['output'] if isinstance(crew_result, dict) else cleaned_content,
                        week,
                        day,
                        platform_name
                    )
                    
                    # Process content according to platform limits
                    processed_content = process_content_for_platform(
                        raw_content, 
                        platform_name,
                        PLATFORM_LIMITS[platform_name]
                    )
                    
                    # Extract title from content
                    title = extract_title_from_content(processed_content)
                    
                    # Calculate word and character counts
                    word_count = len(processed_content.split())
                    char_count = len(processed_content)
                    
                    post = ContentResponse(
                        week_day=f"Week {week} - {day}",
                        title=title,
                        content=processed_content,
                        platform=platform_name,
                        timestamp=datetime.now().isoformat(),
                        word_count=word_count,
                        char_count=char_count
                    )
                    platform_posts.append(post.dict())
            
            results[platform_name] = platform_posts

        # Save to JSON file
        output_file = os.path.join(OUTPUT_DIR, f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Store in database
        try:
            stored_contents = db_manager.store_content(
                content_data=results,
                file_name=file.filename,
                file_type=file_type
            )
            db_storage_status = "success"
            db_storage_message = f"Successfully stored {len(stored_contents)} content items in database"
        except Exception as e:
            db_storage_status = "failed"
            db_storage_message = f"Failed to store in database: {str(e)}"
            print(f"Database storage error: {str(e)}")
        
        return {
            "status": "success",
            "message": "Content generated successfully",
            "output_file": output_file,
            "database_storage": {
                "status": db_storage_status,
                "message": db_storage_message
            },
            "results": results
        }

    except Exception as e:
        print(f"Error during content generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content generation failed: {str(e)}"
        )
    finally:
        # Cleanup uploaded file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

# Add new endpoint to get content from database
@app.get("/get_pending_content")
async def get_pending_content():
    """Get all pending content from database"""
    try:
        pending_content = db_manager.get_pending_content()
        return {
            "status": "success",
            "count": len(pending_content),
            "content": [
                {
                    "id": content.id,
                    "week": content.week,
                    "day": content.day,
                    "title": content.title,
                    "content": content.content,
                    "platform": content.platform.value,
                    "date_upload": content.date_upload.isoformat(),
                    "file_name": content.file_name
                }
                for content in pending_content
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch pending content: {str(e)}"
        )



@app.get("/get_pending_files")
async def get_pending_content():
    """Get all pending files from database"""
    try:
        pending_files = db_manager.get_pending_content()
        
        # Use a dictionary to ensure distinct file names
        unique_files = {}
        for content in pending_files:
            if content.file_name not in unique_files:
                unique_files[content.file_name] = content

        # Convert the unique files into the desired response format
        distinct_files = [
            {
                "date_upload": unique_files[file_name].date_upload.isoformat(),
                "file_name": file_name
            }
            for file_name in unique_files
        ]
        
        return {
            "status": "success",
            "count": len(distinct_files),
            "content": distinct_files
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch pending content: {str(e)}"
        )


# Add new endpoint to get content from database
@app.get("/get_pending_content_file", response_model=dict)
async def get_pending_content(file_name: str = Query(..., description="The name of the file to filter content by")):
    """Get all pending content from a specific file based on file_name."""
    try:
        # Fetch content from the database filtered by file_name
        pending_content = db_manager.get_pending_content_file(file_name=file_name)
        
        if not pending_content:
            raise HTTPException(status_code=404, detail="No content found for the specified file.")

        # Prepare the response
        return {
            "status": "success",
            "count": len(pending_content),
            "content": [
                {
                    "id": content.id,
                    "week": content.week,
                    "day": content.day,
                    "title": content.title,
                    "content": content.content,
                    "platform": content.platform,
                    "date_upload": content.date_upload.isoformat(),
                    "file_name": content.file_name
                }
                for content in pending_content
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch pending content: {str(e)}"
        )



@app.put("/regenerate_script")
async def regenerate_script(content: str):
    """
    Regenerate a script by its content using the script writer agent.
    
    Args:
        content: The content of the content to regenerate
        
    Returns:
        Dict containing the regenerated script details and status
    """
    session = next(db_manager.get_db_session())
    try:
        # Get the existing content
        content = session.query(Content).filter(Content.content== content).first()
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")

        # Create crew with script writer agent
        script_crew = Crew(
            agents=[script_rewriter_agent],
            tasks=[script_rewriter_task],
            process=Process.sequential
        )

        # Generate new script
        crew_result = script_crew.kickoff(
            inputs={
                "text": content.content,
                "day": content.day,
                "week": content.week,
                "platform": content.platform.value,
                "limits": PLATFORM_LIMITS[content.platform.value.lower()]
            }
        )

        # Extract the text content from crew result
        if isinstance(crew_result, dict):
            new_content = str(crew_result.get('output', ''))
        elif hasattr(crew_result, 'raw_output'):
            new_content = str(crew_result.raw_output)
        elif hasattr(crew_result, 'output'):
            new_content = str(crew_result.output)
        else:
            new_content = str(crew_result)

        # Ensure we have content to process
        if not new_content:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate new content"
            )

        # Process the new content according to platform limits
        processed_content = process_content_for_platform(
            new_content,
            content.platform.value.lower(),
            PLATFORM_LIMITS[content.platform.value.lower()]
        )

        # Generate new title
        new_title = extract_title_from_content(processed_content)

        # Update the content in the database
        content.content = processed_content
        content.title = new_title
        session.commit()

        return {
            "status": "success",
            "message": "Script regenerated successfully",
            "content": {
                "id": content.id,
                "week": content.week,
                "day": content.day,
                "title": new_title,
                "content": processed_content,
                "platform": content.platform.value,
                "date_upload": content.date_upload.isoformat(),
                "file_name": content.file_name
            }
        }

    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to regenerate script: {str(e)}"
        )
    finally:
        session.close()




@app.post("/generate_custom_scripts")
async def generate_custom_scripts(
    file: UploadFile = File(...),
    weeks: int = 1,
    days: str = "Monday,Wednesday,Friday",  # Example default value
    platform_posts: str = "instagram:3,facebook:2,twitter:1"  # Example default value
) -> Dict:
    try:
        # Save and extract text from file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get file type
        file_type = Path(file.filename).suffix.lstrip('.')

        # Parse days and platform posts (case-insensitive)
        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        selected_days = []
        for day in days.split(","):
            day_title = day.strip().title()  # Convert to title case for standardization
            if day_title not in valid_days:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid day: {day}. Valid days are: {', '.join(valid_days)}"
                )
            selected_days.append(day_title)

        platform_post_counts = {}
        for platform_info in platform_posts.split(","):
            try:
                platform, count = platform_info.strip().split(":")
                platform_post_counts[platform.lower().strip()] = int(count)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid platform:count format: {platform_info}. Expected format: platform:number"
                )

        processor = FileProcessor()
        try:
            extracted_text = processor.extract_text_from_file(file_path)
            print(f"Successfully extracted text from {file.filename}")
            print(f"Extracted text length: {len(extracted_text)} characters")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file: {str(e)}"
            )
        

                # Prepare inputs dynamically
        research_inputs = {"text": extracted_text}
        if 'file_path' in locals():
            research_inputs['file_path'] = file_path

        # Optional: Add week and day if you want to include them
        research_inputs['week'] = '1'  # Default week
        research_inputs['day'] = selected_days[0] if selected_days else 'Monday' # Default day

        # Research Phase
        research_crew = Crew(
            agents=[script_research_agent],
            tasks=[script_research_task],
            process=Process.sequential
        )
        research_result = research_crew.kickoff(
            inputs={
                "text": extracted_text,
                "file_path": file_path,
                "week": "1",  # Convert to string to match format in extract_content
                "day": "Monday"  # Provide a default day if not already specified
            }
        )
        researched_content = research_result['output'] if isinstance(research_result, dict) else extracted_text


        # QC Phase
        qc_crew = Crew(
            agents=[qc_agent],
            tasks=[qc_task],
            process=Process.sequential
        )
        qc_result = qc_crew.kickoff(
            inputs={
                "text": researched_content
            }
        )

         # Handle CrewOutput correctly
        if hasattr(qc_result, 'output'):
            cleaned_content = qc_result.output
        elif isinstance(qc_result, dict) and 'output' in qc_result:
            cleaned_content = qc_result['output']
        else:
            cleaned_content = researched_content


              #  print("Cleaned Content:", temp_storage)

        # Platform selection (case-insensitive)
        platforms = {
            "linkedin": (linkedin_agent, linkedin_task),
            "instagram": (instagram_agent, instagram_task),
            "facebook": (facebook_agent, facebook_task),
            "twitter": (twitter_agent, twitter_task),
            "wordpress": (wordpress_agent, wordpress_task),
            "youtube": (youtube_agent, youtube_task),
            "tiktok": (tiktok_agent, tiktok_task)
        }

        # Validate platforms (case-insensitive)
        invalid_platforms = [p for p in platform_post_counts.keys() if p not in [k.lower() for k in platforms.keys()]]
        if invalid_platforms:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid platforms: {', '.join(invalid_platforms)}. Available platforms: {', '.join(platforms.keys())}"
            )

         
        # Generate content for each platform
        results = {}
        for platform_name_lower, post_count in platform_post_counts.items():
            platform_name = next(k for k in platforms.keys() if k.lower() == platform_name_lower)
            platform_crew = Crew(
                agents=[platforms[platform_name][0]],
                tasks=[platforms[platform_name][1]],
                process=Process.sequential
            )
            
            platform_posts = []
            for week in range(1, weeks + 1):
                for day in selected_days:
                    # Generate base content for this day
                    crew_result = platform_crew.kickoff(
                        inputs={
                            "text": cleaned_content,
                            "day": day,
                            "week": week,
                            "platform": platform_name,
                            "limits": PLATFORM_LIMITS[platform_name]
                        }
                    )
                    
                    base_content = crew_result['output'] if isinstance(crew_result, dict) else cleaned_content
                    
                    for post_index in range(post_count):
                        # Generate different content for each post using the new function
                        raw_content = generate_different_content(
                            base_content,
                            week,
                            day,
                            platform_name,
                            post_index + 1
                        )
                        
                        # Process content according to platform limits
                        processed_content = process_content_for_platform(
                            raw_content, 
                            platform_name,
                            PLATFORM_LIMITS[platform_name]
                        )
                        
                        # Create unique title for each post
                        title = f"{platform_name} - Week {week}, {day} - Post {post_index + 1}"
                        
                        # Calculate word and character counts
                        word_count = len(processed_content.split())
                        char_count = len(processed_content)
                        
                        post = ContentResponse(
                            week_day=f"Week {week} - {day} - Post {post_index + 1}",
                            title=title,
                            content=processed_content,
                            platform=platform_name,
                            timestamp=datetime.now().isoformat(),
                            word_count=word_count,
                            char_count=char_count
                        )
                        platform_posts.append(post.dict())
            
            results[platform_name] = platform_posts

        # Save to JSON file
        output_file = os.path.join(OUTPUT_DIR, f"custom_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Store in database
        try:
            stored_contents = db_manager.store_content(
                content_data=results,
                file_name=file.filename,
                file_type=file_type
            )
            if stored_contents:  # Ensure some data was actually stored
                db_storage_status = "success"
                db_storage_message = f"Successfully stored {len(stored_contents)} content items in database"
            else:
                db_storage_status = "failed"
                db_storage_message = "No content was stored in the database."

        except Exception as e:
            db_storage_status = "failed"
            db_storage_message = f"Failed to store in database: {str(e)}"
            print(f"Database storage error: {str(e)}")

        
        return {
            "status": "success",
            "message": "Custom content generated successfully",
            "output_file": output_file,
            "database_storage": {
                "status": db_storage_status,
                "message": db_storage_message
            },
            "results": results
        }

    except Exception as e:
        print(f"Error during content generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content generation failed: {str(e)}"
        )
    finally:
        # Cleanup uploaded file
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)   






CACHE_EXPIRATION = 600

class ContentItem(BaseModel):
    type: str
    text: str


class MainContent(BaseModel):
    type: str
    text: str


    
class WeeklyContent(BaseModel):
    week: str
    content_by_days: Dict[str, List[ContentItem]]

class CacheEntry:
    def __init__(self, content: WeeklyContent):
        self.content = content
        self.timestamp = time.time()
        self.temp_id = str(int(self.timestamp))

def cleanup_expired_entries():
    current_time = time.time()
    expired_keys = [
        key for key, entry in temp_storage.items()
        if current_time - entry.timestamp > CACHE_EXPIRATION
    ]
    for key in expired_keys:
        del temp_storage[key]

# Storage for permanent and temporary content
content_storage: Dict[int, WeeklyContent] = {}
temp_storage: Dict[str, CacheEntry] = {}


@app.post("/extract_content")
async def extract_content(
    file: UploadFile = File(...),
    week: int = Form(...),
    days: str = Form(...)
):
    file_path = None
    try:
        if week < 1:
            raise HTTPException(status_code=400, detail="Week must be a positive integer.")
        
        valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        day_list = [day.strip().lower() for day in days.split(",")]
        
        invalid_days = [day for day in day_list if day not in valid_days]
        if invalid_days:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid day(s): {', '.join(invalid_days)}")

        timestamp = int(time.time())
        file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        file_content = await file.read()
        
        # Extract text from PDF
        pdf_text = ""
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
            
        processor = FileProcessor()
        try:
            extracted_text = processor.extract_text_from_file(file_path)
            print(f"Successfully extracted text from {file.filename}")
            print(f"Extracted text length: {len(extracted_text)} characters")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing file: {str(e)}"
            )

        all_weeks_content = {}
        research_crew = Crew(
            agents=[script_research_agent],
            tasks=[script_research_task],
            process=Process.sequential
        )
        
        for current_week in range(1, week + 1):
            week_content = {"content_by_days": {}}
            
            for day in day_list:
                try:
                    research_result = research_crew.kickoff(
                        inputs={
                            "text": extracted_text,
                            "week": current_week,
                            "day": day.capitalize()
                        }
                    )
                    
                    if isinstance(research_result, dict) and 'output' in research_result:
                        day_content = research_result['output']
                        if day_content:
                            week_content["content_by_days"][day.capitalize()] = day_content
                    elif research_result:
                        week_content["content_by_days"][day.capitalize()] = [
                            {"type": "text", "text": str(research_result)}
                        ]
                except Exception as e:
                    print(f"Error processing week {current_week}, {day}: {str(e)}")
                    continue
            
            if week_content["content_by_days"]:
                all_weeks_content[f"Week {current_week}"] = WeeklyContent(
                    week=f"Week {current_week}",
                    content_by_days=week_content["content_by_days"]
                )

        cache_entry = CacheEntry(all_weeks_content)
        temp_storage[cache_entry.temp_id] = cache_entry
        content_storage[int(cache_entry.temp_id)] = all_weeks_content
        cleanup_expired_entries()

        return {
            "status": "success",
            "message": "Content extracted successfully",
            "content": all_weeks_content,
            "temp_id": cache_entry.temp_id,
            "content_storage_key": int(cache_entry.temp_id),
            "timestamp": datetime.now().isoformat(),
            "expiration": datetime.now() + timedelta(seconds=CACHE_EXPIRATION)
        }
    
    except Exception as e:
        print(f"Error during content extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content extraction failed: {str(e)}")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up file: {file_path}")
            except Exception as e:
                print(f"Error cleaning up file: {str(e)}")



                

@app.put("/update_content/{temp_id}")
async def update_content(
    temp_id: str,
    updated_content: WeeklyContent
):
    """Update stored content using temporary ID."""
    try:
        cleanup_expired_entries()
        
        if temp_id not in temp_storage:
            raise HTTPException(
                status_code=404,
                detail="Content not found or has expired. Please extract content again."
            )
        
        temp_storage[temp_id].content = updated_content
        temp_storage[temp_id].timestamp = time.time()
        
        week_str = updated_content.week
        week_num = int(week_str.split()[1]) if len(week_str.split()) > 1 else 0
        if week_num > 0:
            content_storage[week_num] = updated_content
        
        return {
            "status": "success",
            "message": "Content updated successfully",
            "content": updated_content,
            "timestamp": datetime.now().isoformat(),
            "expiration": datetime.now() + timedelta(seconds=CACHE_EXPIRATION)
        }

    except Exception as e:
        print(f"Error during content update: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content update failed: {str(e)}"
        )

@app.get("/temp_content/{temp_id}")
async def get_temp_content(temp_id: str):
    """Get content from temporary storage."""
    cleanup_expired_entries()
    
    if temp_id not in temp_storage:
        raise HTTPException(
            status_code=404,
            detail="Content not found or has expired. Please extract content again."
        )
    
    entry = temp_storage[temp_id]
    return {
        "content": entry.content,
        "timestamp": datetime.fromtimestamp(entry.timestamp).isoformat(),
        "expiration": datetime.fromtimestamp(entry.timestamp + CACHE_EXPIRATION).isoformat()
    }


@app.post("/regenerate_content")
async def regenerate_content(
    week_content: str | None = None,
):
    """Regenerate extracted content using the specified agent and task.
    Accepts only week_content as input.
    """
    try:
        if week_content is None:
            raise HTTPException(
                status_code=400,
                detail="week_content must be provided"
            )
            
        regenerated_content = {}
        
        regenerate_crew = Crew(
            agents=[regenrate_content_agent],
            tasks=[regenrate_content_task],
            process=Process.sequential
        )
        
        try:
            # Create inputs dictionary based on provided parameters
            inputs = {}
            if week_content is not None:
                inputs["week_content"] = week_content
            
            regenerate_result = regenerate_crew.kickoff(inputs=inputs)
            
            if isinstance(regenerate_result, dict) and 'output' in regenerate_result:
                regenerated_content = regenerate_result['output']
            elif regenerate_result:
                regenerated_content = str(regenerate_result)
        except Exception as e:
            print(f"Error regenerating content: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error regenerating content: {str(e)}"
            )
        
        # Create cache entry with only the provided content
        cache_data = {"week_content": week_content}
        new_cache_entry = CacheEntry(cache_data)
        temp_storage[new_cache_entry.temp_id] = new_cache_entry
        content_storage[int(new_cache_entry.temp_id)] = regenerated_content
        
        # Build response with only the relevant fields
        response = {
            "status": "success",
            "message": "Content regenerated successfully",
            # "temp_id": new_cache_entry.temp_id,
            # "content_storage_key": int(new_cache_entry.temp_id),
            # "timestamp": datetime.now().isoformat(),
            # "expiration": datetime.now() + timedelta(seconds=CACHE_EXPIRATION)
        }
        
        # Add the regenerated weekly content to the response
        response["week_content"] = regenerated_content
        
        return response
    
    except Exception as e:
        print(f"Error during content regeneration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Content regeneration failed: {str(e)}"
        )

    


@app.post("/regenerate_subcontent")
async def regenerate_subcontent(
    subcontent: str | None = None
):
    """Regenerate extracted subcontent using the specified agent and task.
    Accepts only subcontent as input.
    """
    try:
        if subcontent is None:
            raise HTTPException(
                status_code=400,
                detail="subcontent must be provided"
            )
            
        regenerated_content = {}
        
        regenerate_crew = Crew(
            agents=[regenrate_content_agent],
            tasks=[regenrate_content_task],
            process=Process.sequential
        )
        
        try:
            # Create inputs dictionary with the provided subcontent
            inputs = {}
            if subcontent is not None:
                inputs["subcontent"] = subcontent
            
            regenerate_result = regenerate_crew.kickoff(inputs=inputs)
            
            if isinstance(regenerate_result, dict) and 'output' in regenerate_result:
                regenerated_content = regenerate_result['output']
            elif regenerate_result:
                regenerated_content = str(regenerate_result)
        except Exception as e:
            print(f"Error regenerating subcontent: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error regenerating subcontent: {str(e)}"
            )
        
        # Create cache entry with the regenerated subcontent
        cache_data = {"subcontent": regenerated_content}
        new_cache_entry = CacheEntry(cache_data)
        temp_storage[new_cache_entry.temp_id] = new_cache_entry
        content_storage[int(new_cache_entry.temp_id)] = regenerated_content
        
        # Build response with the regenerated subcontent
        response = {
            "status": "success",
            "message": "Subcontent regenerated successfully",
            # "temp_id": new_cache_entry.temp_id,
            # "content_storage_key": int(new_cache_entry.temp_id),
            # "timestamp": datetime.now().isoformat(),
            # "expiration": datetime.now() + timedelta(seconds=CACHE_EXPIRATION)
        }
        
        response["subcontent"] = regenerated_content
        
        return response
    
    except Exception as e:
        print(f"Error during subcontent regeneration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Subcontent regeneration failed: {str(e)}"
        )






# Store default values from agents and tasks
DEFAULT_AGENTS = {
    "script_research_agent": {
        "role": script_research_agent.role,
        "goal": script_research_agent.goal,
        "backstory": script_research_agent.backstory
    },
    "qc_agent": {
        "role": qc_agent.role,
        "goal": qc_agent.goal,
        "backstory": qc_agent.backstory
    },
    "script_rewriter_agent": {
        "role": script_rewriter_agent.role,
        "goal": script_rewriter_agent.goal,
        "backstory": script_rewriter_agent.backstory
    },
    "linkedin_agent": {
        "role": linkedin_agent.role,
        "goal": linkedin_agent.goal,
        "backstory": linkedin_agent.backstory
    },
    "instagram_agent": {
        "role": instagram_agent.role,
        "goal": instagram_agent.goal,
        "backstory": instagram_agent.backstory
    },
    "facebook_agent": {
        "role": facebook_agent.role,
        "goal": facebook_agent.goal,
        "backstory": facebook_agent.backstory
    },
    "twitter_agent": {
        "role": twitter_agent.role,
        "goal": twitter_agent.goal,
        "backstory": twitter_agent.backstory
    },
    "wordpress_agent": {
        "role": wordpress_agent.role,
        "goal": wordpress_agent.goal,
        "backstory": wordpress_agent.backstory
    },
    "youtube_agent": {
        "role": youtube_agent.role,
        "goal": youtube_agent.goal,
        "backstory": youtube_agent.backstory
    },
    "tiktok_agent": {
        "role": tiktok_agent.role,
        "goal": tiktok_agent.goal,
        "backstory": tiktok_agent.backstory
    },
    "regenrate_content_agent": {
        "role": regenrate_content_agent.role,
        "goal": regenrate_content_agent.goal,
        "backstory": regenrate_content_agent.backstory
    },
    "regenrate_subcontent_agent": {
        "role": regenrate_subcontent_agent.role,
        "goal": regenrate_subcontent_agent.goal,
        "backstory": regenrate_subcontent_agent.backstory
    },
    
    # Add other agents similarly
}

DEFAULT_TASKS = {
    "script_research_task": {
        "description": script_research_task.description,
        "expected_output": script_research_task.expected_output
    },
    "qc_task": {
        "description": qc_task.description,
        "expected_output": qc_task.expected_output
    },
    "script_rewriter_task": {
        "description": script_rewriter_task.description,
        "expected_output": script_rewriter_task.expected_output
    },
    "linkedin_task": {
        "description": linkedin_task.description,
        "expected_output": linkedin_task.expected_output
    },
    "instagram_task": {
        "description": instagram_task.description,
        "expected_output": instagram_task.expected_output
    },
    "facebook_task": {
        "description": facebook_task.description,
        "expected_output": facebook_task.expected_output
    },
    "twitter_task": {
        "description": twitter_task.description,
        "expected_output": twitter_task.expected_output
    },
    "wordpress_task": {
        "description": wordpress_task.description,
        "expected_output": wordpress_task.expected_output
    },
    "youtube_task": {
        "description": youtube_task.description,
        "expected_output": youtube_task.expected_output
    },
    "tiktok_task": {
        "description": tiktok_task.description,
        "expected_output": tiktok_task.expected_output
    },
    "regenrate_content_task": {
        "description": regenrate_content_task.description,
        "expected_output": regenrate_content_task.expected_output
    },
    "regenrate_subcontent_task": {
        "description": regenrate_subcontent_task.description,
        "expected_output": regenrate_subcontent_task.expected_output
    },
    # Add other tasks similarly
}

# Store current configurations
# Store current configurations
current_agents = DEFAULT_AGENTS.copy()
current_tasks = DEFAULT_TASKS.copy()

class UpdateRequest(BaseModel):
    role: str = None
    goal: str = None
    backstory: str = None
    description: str = None
    expected_output: str = None



def update_python_file(file_path, class_name, updates):
    with open(file_path, "r") as f:
        content = f.read()

    for key, value in updates.items():
        if value:  # Only update if a new value is provided
            pattern = rf'({key}\s*=\s*""")([\s\S]*?)("""\s*)'
            replacement = rf'\1{value}\3'
            content = re.sub(pattern, replacement, content, count=1)  # Replace only first occurrence

    with open(file_path, "w") as f:
        f.write(content)


@app.get("/config/{name}")
def get_config(name: str):
    if name in current_agents:
        return {"current": current_agents[name], "default": DEFAULT_AGENTS[name]}
    elif name in current_tasks:
        return {"current": current_tasks[name], "default": DEFAULT_TASKS[name]}
    else:
        raise HTTPException(status_code=404, detail="Agent or Task not found")

@app.put("/config/{name}")
def update_config(name: str, update: UpdateRequest):
    if name in current_agents:
        agent_file = "agents.py"
        updates = {}
        if update.role:
            current_agents[name]["role"] = update.role
            updates["role"] = update.role
        if update.goal:
            current_agents[name]["goal"] = update.goal
            updates["goal"] = update.goal
        if update.backstory:
            current_agents[name]["backstory"] = update.backstory
            updates["backstory"] = update.backstory
        update_python_file(agent_file, name, updates)
        return {"message": "Agent updated successfully", "current": current_agents[name], "default": DEFAULT_AGENTS[name]}
    elif name in current_tasks:
        task_file = "tasks.py"
        updates = {}
        if update.description:
            current_tasks[name]["description"] = update.description
            updates["description"] = update.description
        if update.expected_output:
            current_tasks[name]["expected_output"] = update.expected_output
            updates["expected_output"] = update.expected_output
        update_python_file(task_file, name, updates)
        return {"message": "Task updated successfully", "current": current_tasks[name], "default": DEFAULT_TASKS[name]}
    else:
        raise HTTPException(status_code=404, detail="Agent or Task not found")

@app.post("/config/reset")
def reset_config():
    global current_agents, current_tasks
    current_agents = {key: value.copy() for key, value in DEFAULT_AGENTS.items()}
    current_tasks = {key: value.copy() for key, value in DEFAULT_TASKS.items()}
    
    def restore_defaults(file_path, defaults):
        with open(file_path, "r") as f:
            content = f.read()

        for class_name, values in defaults.items():
            for key, value in values.items():
                pattern = rf'({key}\s*=\s*""")(.*?)(\s*""")'
                replacement = rf'\1{value}\3'
                content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        with open(file_path, "w") as f:
            f.write(content)

    restore_defaults("agents.py", DEFAULT_AGENTS)
    restore_defaults("tasks.py", DEFAULT_TASKS)

    return {"message": "All configurations reset to default values."}





if __name__ == "__main__":
    import uvicorn
    # Create tables on startup
    db_manager.create_tables()
    uvicorn.run(app, host="0.0.0.0", port=8000)