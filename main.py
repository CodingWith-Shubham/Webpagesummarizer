from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # This will load the .env file

# Fetch API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

# Configure Gemini API
genai.configure(api_key=api_key)

app = FastAPI(title="Web Summarizer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UrlRequest(BaseModel):
    url: str

@app.post("/api/summarize")
async def summarize_webpage(request: UrlRequest):
    try:
        # Fetch webpage content
        response = requests.get(request.url)
        response.raise_for_status()
        
        print(f"Connected successfully! Status: {response.status_code}")
        
        # Parse the webpage
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text content
        text_content = "\n".join([tag.text.strip() for tag in soup.find_all(["p", "div", "span"])])

        # Ensure there is text to summarize
        if not text_content.strip():
            raise HTTPException(status_code=404, detail="No meaningful text found on the webpage.")
        
        # Use Google Gemini API for summarization
        model = genai.GenerativeModel("gemini-1.5-flash")
        ai_response = model.generate_content(f"Summarize this:\n{text_content}")
        
        # Return the summary
        return {"summary": ai_response.text}
    
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching the webpage: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
