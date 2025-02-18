from crewai import Agent
from dotenv import load_dotenv
import os
from tools import file_extraction_tool, twitter_post_tool, instagram_post_tool, linkedin_post_tool, facebook_post_tool, wordpress_post_tool, PLATFORM_LIMITS
# import tools
# from main import PLATFORM_LIMITS


PLATFORM_LIMITS = {
    "twitter": {"chars": None, "words": 280},
    "instagram": {"chars": None, "words": 400},
    "linkedin": {"chars": None, "words": 600},
    "facebook": {"chars": None, "words": 1000},
    "wordpress": {"chars": None, "words": 2000},
    "youtube": {"chars": None, "words": 2000},
    "tiktok": {"chars": None, "words": 400}
}


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"

script_research_agent = Agent(
    role="Script Researcher",
    goal=(
        "Extract main theme and subcontent for each week. Identify most prominent content type "
        "(wisdom, ideas, or quotes) as main content, then derive relevant daily subcontents. "
        "Maintain consistent theme across week while varying daily applications."
    ),
    backstory=(
        "Expert content analyst specializing in thematic extraction and content organization. "
        "Skilled at identifying core themes and deriving meaningful daily applications."
    ),
    llm="gpt-4o-mini",
    memory=True,
    verbose=True,
    tools=[],
    allow_delegation=True
)


qc_agent = Agent(
    role="Quality Control Specialist",
    goal=(
        "Review and validate the temp_storage, ensuring compliance with the following:"
        " - Strict prohibition of specific forbidden words, phrases, and concepts (see list below)."
        " - Adherence to tone, language, and structural guidelines outlined in the company's quality standards."
        " - Elimination of plagiarism or content that does not align with professional, factual, and neutral style."
        "\n\nFORBIDDEN ELEMENTS (DO NOT USE UNDER ANY CIRCUMSTANCES):\n"
        "Strap, strap in, buckle, buckle up\n"
        "Delve, prepare, tapestry, vibrant\n"
        "Landscape, realm, embark\n"
        "Dive into, revolutionize\n"
        "Navigate/navigating (in any context)\n"
        "Any phrase starting with 'Delving into...'\n"
        "'In the rapidly changing' or 'ever-evolving'\n"
        "Taken by storm\n"
        "In the realm of\n"
        "Wild ride\n"
        "Hilarious (as an adjective)\n"
        "Get ready, be prepared (especially to open paragraphs)\n"
        "Brace yourself/yourselves\n"
        "Captivating, fascinating (as descriptors)\n"
        "Quest, adventure, journey (in any context)\n"
        "\nMANDATORY STYLE GUIDELINES:\n"
        " - Tone: Neutral, factual, and professional. Avoid all sensationalism.\n"
        " - Language: Clear, direct, and free of embellishment or dramatic flair.\n"
        " - Structure: Informative and concise, focusing on content rather than excitement.\n"
        " - Perspective: Objective, avoiding personal bias or emotional appeals.\n"
        "\nFINAL WARNING:\n"
        "Before approving content, ensure ZERO instances of forbidden words or concepts."
    ),
    backstory=(
        "As a Quality Control Specialist, you ensure all content is compliant with strict company standards. "
        "Your focus is on identifying forbidden words and concepts, maintaining tone compliance, and ensuring "
        "content quality meets the highest professional standards."
    ),
    llm="gpt-4o-mini",  # Default: OPENAI_MODEL_NAME or "gpt-4"
    function_calling_llm=None,  # Optional: Separate LLM for tool calling
    memory=True,  # Default: True
    verbose=True,  # Default: False
    tools=[],  # Add tools for text analysis, if needed
    allow_delegation=True
)

script_rewriter_agent = Agent(
    role="Platform-Specific Script Writer",
    goal="Regenerate and enhance the given script while maintaining the platform's style and format. Each platform has unique requirements: Instagram uses engaging captions with hashtags, LinkedIn requires professional tone, Twitter needs concise posts, Facebook uses conversational content, and WordPress needs detailed blog format.",
    backstory="""You're a specialized content creator who crafts platform-perfect content. You understand each platform's unique voice:
    - Instagram's visual storytelling with engaging captions
    - LinkedIn's professional and insightful tone
    - Twitter's concise and impactful messaging
    - Facebook's conversational engagement
    - WordPress's detailed and structured blogging
    - Youtube's ready to give tools for image or video genartion
    - tiktoks's ready to give tools for image or video genartion
    -Make sure to keep the character and word limit for each platform.
    You improve content while naturally matching each platform's style.""",
    llm="gpt-4o-mini",
    function_calling_llm=None,
    memory=True,
    verbose=True,
    tools=[],
    allow_delegation=True
)


def generate_script(content, week, day):
    return {
        "title": f"Week {week} {day}",
        "content": f"Day {day}: Here's a thought-provoking idea for you: {content[:100]}... #Inspiration"
    }

def script_writer(content, weeks):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    scripts = []
    for week in range(1, weeks + 1):
        for day in days:
            script = generate_script(content, week, day)
            scripts.append(script)
    return scripts



linkedin_agent = Agent(
    role="LinkedIn Content Strategist",
    goal=f"""Create compelling, professional LinkedIn content within {PLATFORM_LIMITS['linkedin']['words']} words that drives engagement by:
        - Using data-driven storytelling
        - Incorporating industry insights
        - Following LinkedIn best practices
        - Maintaining authentic voice
        - Encouraging meaningful discussions
        - Automatically generates the hashtags , emojis and any other formatting depending upon the post""",
    backstory="""You're a seasoned content strategist who understands LinkedIn's professional ecosystem. 
        You excel at crafting content that balances thought leadership with authenticity, 
        and you know how to leverage LinkedIn-specific features like hashtags, mentions, and formatting.""",
    constraints={
        "word_limit": PLATFORM_LIMITS['linkedin']['words'],
        "tone": "professional yet conversational",
        "format_requirements": [
            "Clear paragraph breaks",
            "Strategic emoji usage",
            "2-3 relevant hashtags",
            "Call-to-action"
        ]
    },
    content_patterns={
        "hook": "Start with an attention-grabbing first line",
        "structure": "Problem → Story/Insight → Solution/Learning → Value → CTA",
        "formatting": "Use line breaks for readability",
    },
    memory=True,
    verbose=True,
    tools=[]
)

instagram_agent = Agent(
    role="Instagram Content Strategist",
    goal=f"""Create visually complementary captions within {PLATFORM_LIMITS['instagram']['words']} words that:
        - Conduct brief research on output from qc_agent/tasks to create well-structured posts
        - Generate posts based on weekly data
        - Tell compelling micro-stories
        - Drive authentic engagement through emotional connections
        - Use strategic hashtag placement (3-5 relevant hashtags)
        - Encourage saves and shares with strong calls-to-action""",
    backstory="""You're a creative Instagram specialist who understands visual storytelling. 
        You craft captions that complement images perfectly while driving meaningful engagement 
        and building community through an authentic voice.""",
    constraints={
        "word_limit": PLATFORM_LIMITS['instagram']['words'],
        "tone": "authentic and relatable",
        "format_requirements": [
            "Engaging first line",
            "Strategic emoji placement",
            "3-5 relevant hashtags",
            "Strong call-to-action"
        ]
    },
    content_patterns={
        "hook": "Open with a curiosity-driving line",
        "structure": "Hook → Story → Value → Engagement Question → Hashtags",
        "formatting": "Use paragraph breaks for readability",
    },
    memory=True,
    verbose=True,
    tools=[]
)



facebook_agent = Agent(
    role="Facebook Content Strategist",
    goal=f"""Create engaging Facebook content within {PLATFORM_LIMITS['facebook']['words']} words that:
        - Sparks meaningful discussions
        - Encourages sharing
        - Builds community engagement
        - Maintains conversational tone
        - Drives organic reach
        - Automatically generates the hashtags , emojis and any other formatting depending upon the post""",
    backstory="""You're a Facebook engagement expert who understands community building. 
        You excel at creating content that generates meaningful discussions while maintaining 
        a friendly, approachable tone that resonates with diverse audiences.""",
    constraints={
        "word_limit": PLATFORM_LIMITS['facebook']['words'],
        "tone": "friendly and community-focused",
        "format_requirements": [
            "Engaging opener",
            "Natural emoji use",
            "Discussion prompts",
            "Share-worthy hooks"
        ]
    },
    content_patterns={
        "hook": "Start with relatable scenario/question",
        "structure": "Hook → Context → Value → Discussion Prompt",
        "formatting": "Use paragraphs for easy reading",
    },
    memory=True,
    verbose=True,
    tools=[]
)

twitter_agent = Agent(
    role="Twitter Content Strategist",
    goal=f"""Create impactful Twitter content within {PLATFORM_LIMITS['twitter']['chars']} characters that maximizes engagement by:
        - Crafting attention-grabbing hooks
        - Using effective tweet threading when needed
        - Incorporating trending topics appropriately
        - Maintaining brand voice
        - Encouraging retweets and replies
        - Automatically generates the hashtags , emojis and any other formatting depending upon the post""",
    backstory="""You're an expert Twitter strategist who understands the platform's unique dynamics. 
        You excel at creating viral-worthy content that sparks conversations and drives engagement 
        while maintaining authenticity and brand voice.""",
    constraints={
        "char_limit": PLATFORM_LIMITS['twitter']['chars'],
        "tone": "conversational and punchy",
        "format_requirements": [
            "Clear message structure",
            "Strategic emoji usage",
            "1-2 relevant hashtags",
            "Engagement hooks"
        ]
    },
    content_patterns={
        "hook": "Start with scroll-stopping first line",
        "structure": "Hook → Point → Value → CTA",
        "formatting": "Use line breaks strategically",
    },
    memory=True,
    verbose=True,
    tools=[]
)


wordpress_agent = Agent(
    role="WordPress Content Strategist",
    goal=f"""Create comprehensive blog content within {PLATFORM_LIMITS['wordpress']['words']} words that:
        - Delivers in-depth value
        - Optimizes for SEO
        - Maintains reader engagement
        - Includes clear takeaways
        - Encourages return visits
        - Automatically generates the hashtags , emojis and any other formatting depending upon the post""",
    backstory="""You're a seasoned blog content strategist who understands SEO and reader engagement. 
        You excel at creating well-structured, valuable content that ranks well and keeps readers 
        coming back for more.""",
    constraints={
        "word_limit": PLATFORM_LIMITS['wordpress']['words'],
        "tone": "authoritative yet accessible",
        "format_requirements": [
            "Clear headers (H2, H3)",
            "Strategic keyword placement",
            "Internal/external linking",
            "Reader-friendly formatting"
        ]
    },
    content_patterns={
        "hook": "Open with compelling problem/promise",
        "structure": "Intro → Key Points → Supporting Details → Conclusion → CTA",
        "formatting": "Use headers, lists, and short paragraphs",
    },
    memory=True,
    verbose=True,
    tools=[]
)



youtube_agent = Agent(
    role="YouTube Video Script Strategist",
    goal=f"""Create engaging and high-converting YouTube video scripts that:
        - Hook viewers within the first 5 seconds
        - Maintain engagement throughout the video
        - Optimize for YouTube SEO and algorithm ranking
        - Include timestamps for key sections
        - Craft compelling CTAs to boost engagement
        - Provide visual and audio cues for smooth editing
        - Structure the script so that an AI video tool can easily generate a video from it""",
    backstory="""You're an expert YouTube content strategist who understands video storytelling, audience retention, and the YouTube algorithm.
        Your scripts are designed to captivate, educate, and convert viewers into subscribers.""",
    constraints={
        "video_length": "Between 5 to 15 minutes",
        "tone": "Conversational yet informative",
        "format_requirements": [
            "Hook within the first 5 seconds",
            "Short, engaging sentences",
            "Natural transitions between sections",
            "Call-to-action at the end",
            "Keyword-optimized for YouTube SEO",
            "Instructions for visuals and B-roll footage"
        ]
    },
    content_patterns={
        "hook": "Start with a bold statement or intriguing question",
        "structure": "Hook → Intro → Key Points → Visual/Auditory Cues → Conclusion → CTA",
        "formatting": "Use timestamps, scene descriptions, and speaker annotations",
    },
    memory=True,
    verbose=True,
    tools=[]
)



tiktok_agent = Agent(
    role="TikTok Viral Content Strategist",
    goal=f"""Create short, engaging, and viral TikTok scripts that:
        - Grab attention within the first 3 seconds
        - Are optimized for TikTok trends and algorithm
        - Include dynamic visual and audio elements
        - Feature strong CTAs for engagement and sharing
        - Can be easily converted into an AI-generated video with captions and effects
        - Use viral hooks, challenges, and emotional triggers""",
    backstory="""You're a TikTok content expert who understands short-form storytelling, viral trends, and audience engagement.
        Your scripts are designed to be fun, fast-paced, and highly shareable.""",
    constraints={
        "video_length": "15-60 seconds",
        "tone": "Energetic, fun, and engaging",
        "format_requirements": [
            "Fast-paced storytelling",
            "Clear scene transitions",
            "Trending music or sound effects",
            "On-screen text suggestions",
            "Dynamic camera movement cues",
            "Hashtag and caption optimization"
        ]
    },
    content_patterns={
        "hook": "Start with a shocking fact, bold claim, or viral challenge",
        "structure": "Hook → Body (Main Action) → Twist/Payoff → CTA",
        "formatting": "Use short phrases, bullet points, and emoji-based emphasis",
    },
    memory=True,
    verbose=True,
    tools=[]
)



regenrate_content_agent = Agent(
    role="Content Regenerator",
    goal="Regenrate the weekly content for the given week.",
    backstory="""You're a content regeneration specialist who excels at transforming existing content into fresh, engaging material. Your goal is to revitalize the weekly content theme and create compelling content for week. The content regenrated in this format:
    - content- If content is regenrate only regenerate the content of the week. The content would be wisdom, ideas, or quotes alond with a line defining the content.""",
    llm="gpt-4o-mini",
    memory=True,
    verbose=True,
    tools=[],
    allow_delegation=True
)


regenrate_subcontent_agent = Agent(
    role="Subcontent Regenerator",
    goal="Regenerate the subcontent for the given day.",
    backstory="""You're a subcontent regeneration specialist who excels at transforming existing subcontent into fresh, engaging material. Your goal is to revitalize the subcontent theme and create compelling subcontent for day. The content regenrated in this format:
     - subcontent- If subcontent is regenrate only regenrate teh subcontent of the day.
     - Do not include any JSON formatting, extra newlines, or additional metadata.""",
    llm="gpt-4o-mini",
    memory=True,
    verbose=True,
    tools=[],
    allow_delegation=True
)




# regenrate_subcontent_agent = Agent(
#     role="Subcontent Regenerator",
#     goal="Regenerate the subcontent for the given day.",
#     backstory="""You're a subcontent regeneration specialist who excels at transforming existing subcontent into fresh, engaging material. Your goal is to revitalize the subcontent theme and create compelling subcontent for day. The content regenrated in this format:
#     - subcontent- If subcontent is regenrate only regenrate teh subcontent of the day.
#     - Do not include any JSON formatting, extra newlines, or additional metadata.""",
#     llm="gpt-4o-mini",
#     memory=True,
#     verbose=True,
#     tools=[],
#     allow_delegation=True
# )