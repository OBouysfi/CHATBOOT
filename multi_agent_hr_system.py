import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from bs4 import BeautifulSoup
import sqlite3
from pathlib import Path

# Core imports for your existing RAG system
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from get_embedding_function import get_embedding_function

import google.generativeai as genai
import os
from typing import Dict, Any

# Configuration
CHROMA_PATH = "./chroma"
DATABASE_PATH = "./hr_knowledge.db"
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    RAG_SPECIALIST = "rag_specialist"  
    WEB_RESEARCHER = "web_researcher"
    LABOR_LAW_EXPERT = "labor_law_expert"
    RECRUITMENT_SPECIALIST = "recruitment_specialist"
    PAYROLL_CALCULATOR = "payroll_calculator"
    POLICY_ADVISOR = "policy_advisor"
    PERFORMANCE_ANALYST = "performance_analyst"

@dataclass
class AgentPerformance:
    agent_type: AgentType
    processing_time: float
    timestamp: datetime

@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    confidence: float
    sources: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = None
    performance: List[AgentPerformance] = None

class MoroccanHRKnowledgeBase:
    """Enhanced knowledge base for Moroccan HR information"""
    
    def __init__(self):
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for caching web research"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS web_cache (
                id INTEGER PRIMARY KEY,
                query TEXT,
                content TEXT,
                source_url TEXT,
                timestamp DATETIME,
                agent_type TEXT,
                is_valid BOOLEAN DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS moroccan_hr_laws (
                id INTEGER PRIMARY KEY,
                law_code TEXT,
                title TEXT,
                description TEXT,
                last_updated DATE,
                source_url TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def cache_web_result(self, query: str, content: str, source_url: str, agent_type: str):
        """Cache web research results"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO web_cache (query, content, source_url, timestamp, agent_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (query, content, source_url, datetime.now(), agent_type))
        
        conn.commit()
        conn.close()
    
    def get_cached_result(self, query: str, agent_type: str) -> Optional[str]:
        """Retrieve cached results if still valid"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(seconds=CACHE_DURATION)
        
        cursor.execute('''
            SELECT content FROM web_cache 
            WHERE query = ? AND agent_type = ? AND timestamp > ? AND is_valid = 1
            ORDER BY timestamp DESC LIMIT 1
        ''', (query, agent_type, cutoff_time))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None

class BaseAgent:
    """Base class for all HR agents using Gemini API"""

    def __init__(self, agent_type: AgentType, model_name: str = "gemini-2.0-flash"):
        self.agent_type = agent_type
        self.model_name = model_name
        self.knowledge_base = MoroccanHRKnowledgeBase()
        # self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        
        # Configure Gemini
        genai.configure(api_key="AIzaSyC92nou-dcgVOiAzLBYl6hQb3SSaMpbNxs")
        self.model = genai.GenerativeModel(model_name=self.model_name)

    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Must be implemented by child agents"""
        raise NotImplementedError("Each agent must implement process method")

    def generate_response(self, prompt: str) -> str:
        """Call Gemini API with prompt and return text content"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error in Gemini API call: {e}")
            return "Error generating response."

    def log_interaction(self, query: str, response: str):
        """Log agent interactions for debugging"""
        logging.info(f"[{self.agent_type.value}] Query: {query[:100]}...")
        logging.info(f"[{self.agent_type.value}] Response: {response[:200]}...")

class OrchestratorAgent(BaseAgent):
    """Main orchestrator that routes queries to appropriate specialized agents"""
    
    def __init__(self):
        super().__init__(AgentType.ORCHESTRATOR)
        self.agents = {}
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize all specialized agents"""
        self.agents = {
            AgentType.RAG_SPECIALIST: RAGSpecialistAgent(),
            AgentType.WEB_RESEARCHER: WebResearcherAgent(),
            AgentType.LABOR_LAW_EXPERT: LaborLawExpertAgent(),
            AgentType.RECRUITMENT_SPECIALIST: RecruitmentSpecialistAgent(),
            AgentType.PAYROLL_CALCULATOR: PayrollCalculatorAgent(),
            AgentType.POLICY_ADVISOR: PolicyAdvisorAgent(),
            AgentType.PERFORMANCE_ANALYST: PerformanceAnalystAgent()
        }
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Route query to appropriate agents and synthesize response with performance tracking"""
        # Analyze query to determine which agents to involve
        agent_routing = await self.analyze_query_routing(query)
        
        # Collect responses from relevant agents with performance tracking
        agent_responses = []
        processing_steps = []
        agent_performances = []
        
        for agent_type in agent_routing:
            if agent_type in self.agents:
                try:
                    start_time = datetime.now()
                    response = await self.agents[agent_type].process(query, context)
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    agent_responses.append(response)
                    
                    # Record performance data
                    agent_perf = AgentPerformance(
                        agent_type=agent_type,
                        processing_time=processing_time,
                        timestamp=end_time
                    )
                    agent_performances.append(agent_perf)
                    
                    # Format step information
                    step_info = (
                        f"{agent_type.value}: {processing_time:.2f}s "
                        f"({end_time.strftime('%H:%M:%S')})"
                    )
                    processing_steps.append(step_info)
                    
                except Exception as e:
                    logging.error(f"Error in {agent_type.value}: {str(e)}")
                    processing_steps.append(f"{agent_type.value}: failed ({datetime.now().strftime('%H:%M:%S')})")
        
        # Synthesize final response
        final_response = await self.synthesize_responses(query, agent_responses)
        
        # Flatten sources list
        all_sources = []
        for resp in agent_responses:
            all_sources.extend(resp.sources)
        
        # Format processing steps for metadata
        formatted_steps = "\n".join(processing_steps)
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=final_response,
            confidence=0.9,
            sources=all_sources,
            timestamp=datetime.now(),
            metadata={
                "involved_agents": [resp.agent_type.value for resp in agent_responses],
                "processing_steps": formatted_steps
            },
            performance=agent_performances
        )
    
    def format_processing_steps(self, steps: List[str]) -> str:
        """Format processing steps for display"""
        return "\n".join([f"• {step}" for step in steps])
    
    async def analyze_query_routing(self, query: str) -> List[AgentType]:
        """Determine which agents should handle the query"""
        
        routing_prompt = f"""
        Analyze this HR query for Morocco and determine which specialized agents should handle it.
        
        Query: {query}
        
        Available agents:
        - RAG_SPECIALIST: Internal documents and policies
        - WEB_RESEARCHER: Current web information about Moroccan HR
        - LABOR_LAW_EXPERT: Moroccan labor law and regulations
        - RECRUITMENT_SPECIALIST: Hiring, interviewing, talent acquisition
        - PAYROLL_CALCULATOR: Salary calculations, benefits, taxes
        - POLICY_ADVISOR: HR policies and procedures
        - PERFORMANCE_ANALYST: Performance management and analytics
        
        Return only the agent names that should be involved, separated by commas.
        """
        
        response_text = self.generate_response(routing_prompt)
        
        # Parse agent types from response
        agent_types = []
        for agent_name in response_text.split(','):
            agent_name = agent_name.strip().upper()
            try:
                agent_types.append(AgentType[agent_name])
            except KeyError:
                continue
        
        # Always include RAG specialist for internal knowledge
        if AgentType.RAG_SPECIALIST not in agent_types:
            agent_types.append(AgentType.RAG_SPECIALIST)
            
        return agent_types
    
    async def synthesize_responses(self, query: str, responses: List[AgentResponse]) -> str:
        """Combine multiple agent responses into coherent answer"""
        
        if not responses:
            return "Je n'ai pas pu trouver d'informations pertinentes pour votre question."
        
        synthesis_prompt = f"""
        Vous êtes un assistant RH expert pour le Maroc. Synthétisez les réponses suivantes en une réponse cohérente et complète.
        
        Question originale: {query}
        
        Réponses des agents spécialisés:
        """
        
        for resp in responses:
            synthesis_prompt += f"\n\n--- Agent {resp.agent_type.value} ---\n{resp.content}"
        
        synthesis_prompt += """
        
        Instructions:
        1. Créez une réponse unifiée en français
        2. Intégrez les informations de tous les agents
        3. Résolvez les contradictions en privilégiant les sources les plus fiables
        4. Mentionnez les sources importantes
        5. Soyez spécifique au contexte marocain
        """
        
        return self.generate_response(synthesis_prompt)

class RAGSpecialistAgent(BaseAgent):
    """Enhanced RAG agent using your existing vector store"""
    
    def __init__(self):
        super().__init__(AgentType.RAG_SPECIALIST)
        self.setup_rag()
    
    def setup_rag(self):
        """Initialize RAG components"""
        embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        En tant qu'expert RH au Maroc, répondez à la question en utilisant uniquement le contexte fourni.
        
        Contexte des documents internes:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Répondez en français
        2. Basez-vous uniquement sur le contexte fourni
        3. Si l'information n'est pas dans le contexte, dites-le clairement
        4. Citez les sources pertinentes
        5. Adaptez au contexte RH marocain
        
        Réponse:
        """)
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Process query using RAG system"""
        start_time = datetime.now()
        
        # Retrieve relevant documents
        results = self.db.similarity_search_with_score(query, k=5)
        
        if not results:
            end_time = datetime.now()
            return AgentResponse(
                agent_type=self.agent_type,
                content="Aucun document interne pertinent trouvé.",
                confidence=0.1,
                sources=[],
                timestamp=end_time,
                performance=[AgentPerformance(
                    agent_type=self.agent_type,
                    processing_time=(end_time - start_time).total_seconds(),
                    timestamp=end_time
                )]
            )
        
        # Prepare context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = self.prompt_template.format(context=context_text, question=query)
        
        # Generate response
        response_text = self.generate_response(prompt)
        end_time = datetime.now()
        
        # Extract sources
        sources = []
        for doc, _score in results:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response_text,
            confidence=0.8,
            sources=sources,
            timestamp=end_time,
            metadata={"similarity_scores": [score for _, score in results]},
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class WebResearcherAgent(BaseAgent):
    """Agent for researching current HR information from web"""
    
    def __init__(self):
        super().__init__(AgentType.WEB_RESEARCHER)
        self.search_tool = DuckDuckGoSearchRun()
        self.moroccan_hr_sites = [
            "emploi.ma",
            "rekrute.com",
            "emploi-public.ma",
            "anapec.org",
            "cnss.ma",
            "service-public.ma",
            "cnops.org.ma",
            "marocannonces.com",
            "jobi.ma",
            "novojob.com/maroc",
            "diorh.com",
            "ma.indeed.com",
            "africsearch.com",
            "groupeiscae.ma",
        ]
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Research current HR information from web"""
        start_time = datetime.now()
        
        # Check cache first
        cached_result = self.knowledge_base.get_cached_result(query, self.agent_type.value)
        if cached_result:
            end_time = datetime.now()
            return AgentResponse(
                agent_type=self.agent_type,
                content=cached_result,
                confidence=0.7,
                sources=["Cache"],
                timestamp=end_time,
                metadata={"from_cache": True},
                performance=[AgentPerformance(
                    agent_type=self.agent_type,
                    processing_time=(end_time - start_time).total_seconds(),
                    timestamp=end_time
                )]
            )
        
        # Prepare Morocco-specific search queries
        morocco_queries = [
            f"{query} Maroc RH",
            f"{query} ressources humaines Maroc",
            f"{query} emploi Maroc 2024",
            f"{query} droit travail Maroc"
        ]
        
        research_results = []
        
        for search_query in morocco_queries[:2]:  # Limit to 2 searches
            try:
                search_results = self.search_tool.run(search_query)
                research_results.append(search_results)
            except Exception as e:
                logging.error(f"Search error: {str(e)}")
        
        if not research_results:
            end_time = datetime.now()
            return AgentResponse(
                agent_type=self.agent_type,
                content="Impossible de récupérer des informations actuelles du web.",
                confidence=0.1,
                sources=[],
                timestamp=end_time,
                performance=[AgentPerformance(
                    agent_type=self.agent_type,
                    processing_time=(end_time - start_time).total_seconds(),
                    timestamp=end_time
                )]
            )
        
        # Synthesize web research
        synthesis_prompt = f"""
        Synthétisez les informations web suivantes pour répondre à cette question RH au Maroc:
        
        Question: {query}
        
        Résultats de recherche:
        {' '.join(research_results)}
        
        Instructions:
        1. Répondez en français
        2. Concentrez-vous sur le contexte marocain
        3. Mentionnez les informations les plus récentes
        4. Soyez factuel et précis
        """
        
        response = self.generate_response(synthesis_prompt)
        end_time = datetime.now()
        
        # Cache the result
        self.knowledge_base.cache_web_result(query, response, "web_search", self.agent_type.value)
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response,
            confidence=0.7,
            sources=["Recherche web"],
            timestamp=end_time,
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class LaborLawExpertAgent(BaseAgent):
    """Expert in Moroccan labor law and regulations"""
    
    def __init__(self):
        super().__init__(AgentType.LABOR_LAW_EXPERT)
        self.law_knowledge = self.load_moroccan_labor_laws()
    
    def load_moroccan_labor_laws(self) -> Dict[str, str]:
        """Load key Moroccan labor law references"""
        return {
            "code_travail": "Code du travail marocain - Dahir n° 1-03-194",
            "duree_travail": "Durée légale: 44h/semaine, 8h/jour",
            "conges_payes": "18 jours ouvrables minimum par an",
            "salaire_minimum": "SMIG et SMAG selon secteur",
            "preavis": "Préavis selon ancienneté et qualification",
            "indemnites": "Indemnités de licenciement selon ancienneté"
        }
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Provide expert advice on Moroccan labor law"""
        start_time = datetime.now()
        
        law_prompt = f"""
        En tant qu'expert en droit du travail marocain, répondez à cette question:
        
        Question: {query}
        
        Références légales disponibles:
        {json.dumps(self.law_knowledge, indent=2, ensure_ascii=False)}
        
        Instructions:
        1. Citez les articles de loi pertinents
        2. Expliquez les implications pratiques
        3. Mentionnez les exceptions si applicables
        4. Donnez des conseils de conformité
        5. Répondez en français
        
        Si vous n'êtes pas certain d'une information légale, recommandez de consulter un avocat spécialisé.
        """
        
        response = self.generate_response(law_prompt)
        end_time = datetime.now()
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response,
            confidence=0.85,
            sources=["Code du travail marocain", "Dahir n° 1-03-194"],
            timestamp=end_time,
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class RecruitmentSpecialistAgent(BaseAgent):
    """Specialist in recruitment and talent acquisition for Morocco"""
    def __init__(self):
        super().__init__(AgentType.RECRUITMENT_SPECIALIST)
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Provide recruitment expertise"""
        start_time = datetime.now()
        
        recruitment_prompt = f"""
        En tant que spécialiste du recrutement au Maroc, répondez à cette question:
        
        Question: {query}
        
        Considérez:
        - Le marché de l'emploi marocain
        - Les pratiques culturelles locales
        - Les canaux de recrutement populaires
        - Les compétences recherchées
        - Les défis du recrutement au Maroc
        
        Fournissez des conseils pratiques et spécifiques au contexte marocain.
        """
        
        response = self.generate_response(recruitment_prompt)
        end_time = datetime.now()
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response,
            confidence=0.8,
            sources=["Expertise recrutement Maroc"],
            timestamp=end_time,
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class PayrollCalculatorAgent(BaseAgent):
    """Specialist in Moroccan payroll calculations and benefits"""
    
    def __init__(self):
        super().__init__(AgentType.PAYROLL_CALCULATOR)
        self.tax_rates = {
            "ir": {"0-30000": 0, "30001-50000": 0.10, "50001-60000": 0.20, "60001-80000": 0.30, "80001+": 0.38},
            "cnss": 0.0426,  # Employee contribution
            "cnss_employer": 0.2075,  # Employer contribution
            "mut": 0.0225,  # AMO
            "taxe_professionnelle": 0.30
        }
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Calculate payroll and provide salary information"""
        start_time = datetime.now()
        
        payroll_prompt = f"""
        En tant qu'expert en paie au Maroc, répondez à cette question:
        
        Question: {query}
        
        Taux et informations disponibles:
        - Impôt sur le revenu: Barème progressif
        - CNSS salarié: 4.26%
        - CNSS employeur: 20.75%
        - AMO: 2.25%
        - Taxe professionnelle: 30% sur IR
        
        Instructions:
        1. Fournissez des calculs précis si demandés
        2. Expliquez les composantes de la paie marocaine
        3. Mentionnez les obligations légales
        4. Donnez des exemples concrets
        """
        
        response = self.generate_response(payroll_prompt)
        end_time = datetime.now()
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response,
            confidence=0.9,
            sources=["Réglementation paie Maroc", "Barème IR"],
            timestamp=end_time,
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class PolicyAdvisorAgent(BaseAgent):
    """Advisor for HR policies and procedures"""
    def __init__(self):
        super().__init__(AgentType.POLICY_ADVISOR)
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Provide HR policy guidance"""
        start_time = datetime.now()
        
        policy_prompt = f"""
        En tant que conseiller en politiques RH au Maroc, répondez à cette question:
        
        Question: {query}
        
        Considérez:
        - Les meilleures pratiques RH
        - La conformité légale marocaine
        - La culture d'entreprise locale
        - L'équilibre employé-employeur
        - Les tendances RH modernes
        
        Proposez des politiques pratiques et adaptées au contexte marocain.
        """
        
        response = self.generate_response(policy_prompt)
        end_time = datetime.now()
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response,
            confidence=0.8,
            sources=["Meilleures pratiques RH"],
            timestamp=end_time,
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class PerformanceAnalystAgent(BaseAgent):
    """Analyst for performance management and HR analytics"""
    def __init__(self):
        super().__init__(AgentType.PERFORMANCE_ANALYST)
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Provide performance management insights"""
        start_time = datetime.now()
        
        performance_prompt = f"""
        En tant qu'analyste de performance RH au Maroc, répondez à cette question:
        
        Question: {query}
        
        Considérez:
        - Les KPIs RH pertinents
        - Les méthodes d'évaluation
        - Les outils d'analyse
        - Les benchmarks du marché marocain
        - Les stratégies d'amélioration
        
        Fournissez des insights basés sur les données et les meilleures pratiques.
        """
        
        response = self.generate_response(performance_prompt)
        end_time = datetime.now()
        
        return AgentResponse(
            agent_type=self.agent_type,
            content=response,
            confidence=0.8,
            sources=["Analytics RH", "Benchmarks marché"],
            timestamp=end_time,
            performance=[AgentPerformance(
                agent_type=self.agent_type,
                processing_time=(end_time - start_time).total_seconds(),
                timestamp=end_time
            )]
        )

class MoroccanHRAssistant:
    """Main interface for the multi-agent HR system"""
    
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hr_assistant.log'),
                logging.StreamHandler()
            ]
        )
    
    async def ask(self, question: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Main interface for asking questions - returns full AgentResponse object"""
        try:
            response = await self.orchestrator.process(question, context)
            return response
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return AgentResponse(
                agent_type=AgentType.ORCHESTRATOR,
                content=f"Désolé, une erreur s'est produite: {str(e)}",
                confidence=0.0,
                sources=[],
                timestamp=datetime.now(),
                metadata={
                    "involved_agents": [],
                    "processing_steps": "Erreur lors du traitement",
                    "error": str(e)
                }
            )
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {
            agent_type.value: "Active" 
            for agent_type in self.orchestrator.agents.keys()
        }

# CLI Interface
# CLI Interface Update for your backend
async def main():
    """Command line interface for the HR assistant"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Assistant RH Multi-Agents pour le Maroc")
    parser.add_argument("--question", "-q", type=str, help="Question à poser")
    parser.add_argument("--interactive", "-i", action="store_true", help="Mode interactif")
    
    args = parser.parse_args()
    
    assistant = MoroccanHRAssistant()
    
    if args.interactive:
        print("🇲🇦 Assistant RH Multi-Agents pour le Maroc")
        print("Tapez 'quit' pour quitter, 'status' pour voir l'état des agents")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Votre question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            elif question.lower() == 'status':
                status = assistant.get_agent_status()
                print("\n📊 État des agents:")
                for agent, state in status.items():
                    print(f"  • {agent}: {state}")
                continue
            
            if question:
                print("\n🤖 Traitement en cours...")
                response = await assistant.ask(question)
                print(f"\n✅ Réponse:\n{response.content}")
                if hasattr(response, 'metadata') and response.metadata:
                    steps = response.metadata.get("processing_steps", "Aucune étape disponible")
                    print(f"\n🔄 Étapes de traitement:\n{steps}")
    
    elif args.question:
        response = await assistant.ask(args.question)
        print(response.content)
        if hasattr(response, 'metadata') and response.metadata:
            steps = response.metadata.get("processing_steps", "Aucune étape disponible")
            print(f"\n🔄 Étapes de traitement:\n{steps}")
    else:
        print("Utilisez --question 'votre question' ou --interactive pour le mode interactif")

if __name__ == "__main__":
    asyncio.run(main())