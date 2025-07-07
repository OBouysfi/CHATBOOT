from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import uvicorn
import time
import logging
from multi_agent_hr_system import MoroccanHRAssistant, AgentResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hr_assistant_api.log'),
        logging.StreamHandler()
    ]
)

app = FastAPI(
    title="Moroccan HR Assistant API",
    description="Multi-Agent HR Assistant System for Morocco",
    version="1.0.0"
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = MoroccanHRAssistant()

class QuestionRequest(BaseModel):
    question: str
    context: dict = None

class MetadataResponse(BaseModel):
    involved_agents: list
    processing_time: float
    confidence: float
    sources: list
    timestamp: str = None

class AssistantResponse(BaseModel):
    content: str
    metadata: MetadataResponse

class AgentStatusResponse(BaseModel):
    agents: dict

@app.post("/ask", summary="Ask a question to the HR assistant", response_model=AssistantResponse)
async def ask_question(request: QuestionRequest):
    """Submit HR-related questions about Morocco"""
    start_time = time.time()
    try:
        response = await assistant.ask(request.question, request.context)
        processing_time = time.time() - start_time
        
        # Debug logging
        logging.info(f"Response type: {type(response)}")
        logging.info(f"Response content: {response}")
        
        # Handle AgentResponse object
        if hasattr(response, 'content'):
            # Extract involved agents from metadata
            involved_agents = []
            if hasattr(response, 'metadata') and response.metadata:
                involved_agents = response.metadata.get("involved_agents", [])
            
            # If no agents specified, set default
            if not involved_agents:
                involved_agents = ["general"]
            
            # Extract sources
            sources = getattr(response, 'sources', [])
            
            # Extract confidence
            confidence = getattr(response, 'confidence', 0.5)
            
            return {
                "content": response.content,
                "metadata": {
                    "involved_agents": involved_agents,
                    "processing_time": processing_time,
                    "confidence": confidence,
                    "sources": sources,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        else:
            # Fallback for unexpected response format
            logging.error(f"Unexpected response format: {type(response)}")
            return {
                "content": str(response),
                "metadata": {
                    "involved_agents": ["error"],
                    "processing_time": processing_time,
                    "confidence": 0.0,
                    "sources": [],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
    except Exception as e:
        logging.error(f"Error processing question: {str(e)}")
        logging.error(f"Exception type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents", response_model=AgentStatusResponse, summary="Get agent status")
async def get_agent_status():
    """Check availability of all agents"""
    try:
        return {"agents": assistant.get_agent_status()}
    except Exception as e:
        logging.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)