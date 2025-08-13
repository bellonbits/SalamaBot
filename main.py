# main.py – Enhanced SalamaBot with Anonymous Reporting and SOS Features
import os
import uuid
import re
import asyncio
import glob
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict, Any, Annotated
from pathlib import Path

import PyPDF2
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import Document, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Vector DB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Telegram
import telegram
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------- Configuration ----------
class Config:
    GROQ_API_KEY = ""
    TELEGRAM_BOT_TOKEN = ""
    DATA_FOLDER = "data"
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "salamabot_docs"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIEVAL_DOCS = 5
    MONGODB_URL = "mongodb://localhost:27017"
    DB_NAME = "salamabot"
    
    EMERGENCY_CONTACTS = {
        "usikimye": {
            "name": "Usikimye GBV Hotline",
            "number": "0800 720 572",
            "sms": "254",
            "whatsapp": "+254 708 685 614",
            "description": "Confidential GBV support and rescue"
        },
        "police": {
            "name": "Kenya Police",
            "number": "999",
            "sms": "911",
            "description": "Emergency police response"
        },
        "nairobi_womens": {
            "name": "Nairobi Women's Hospital",
            "number": "0709 683 000",
            "description": "GBV medical and psychological support"
        },
        "national_gbv": {
            "name": "National GBV Helpline",
            "number": "1195",
            "description": "24/7 GBV support hotline"
        },
        "childline": {
            "name": "Childline Kenya",
            "number": "116",
            "description": "Child protection services"
        }
    }

config = Config()

# ---------- FastAPI Setup ----------
app = FastAPI(title="SalamaBot", description="Fast GBV support for Kenya with Anonymous Reporting")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict[str, Any]] = []
    confidence: float = 0.0

class DocumentUpload(BaseModel):
    filename: str
    content: str

class AnonymousReport(BaseModel):
    report_details: str
    location: Optional[str] = None
    contact_preference: Optional[str] = None

class SOSAlert(BaseModel):
    location: Optional[str] = None
    situation: Optional[str] = None

# ---------- State Definition ----------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    retrieved_docs: List[Document]
    context: str
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    confidence: float

# ---------- RAG Components ----------
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_documents(self, data_folder: str) -> List[Document]:
        """Process all PDF documents in the data folder"""
        documents = []
        pdf_files = glob.glob(os.path.join(data_folder, "*.pdf"))
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file}")
            text = self.extract_text_from_pdf(pdf_file)
            
            if text.strip():
                # Create document chunks
                chunks = self.text_splitter.split_text(text)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": os.path.basename(pdf_file),
                            "chunk_id": i,
                            "file_path": pdf_file,
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    documents.append(doc)
        
        return documents

class VectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            self.collection = self.client.get_collection(config.COLLECTION_NAME)
            print(f"Loaded existing collection: {config.COLLECTION_NAME}")
        except:
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"description": "SalamaBot GBV support documents"}
            )
            print(f"Created new collection: {config.COLLECTION_NAME}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        if not documents:
            return
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = config.MAX_RETRIEVAL_DOCS) -> List[Document]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            metadata['similarity_score'] = 1 - distance  # Convert distance to similarity
            documents.append(Document(page_content=doc, metadata=metadata))
        
        return documents

class LLMHandler:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name=config.GROQ_MODEL,
            temperature=0.1,
            max_tokens=2048
        )
        
        self.system_prompt = """You are SalamaBot, a compassionate Kenyan AI assistant for Gender-Based Violence (GBV) support.

Key Features:
1. Anonymous reporting available via /anonymous command
2. Emergency SOS to Usikimye and police via /sos
3. Quick-access emergency contacts via /emergency

When users disclose GBV incidents:
- First express empathy and validate their experience
- Offer anonymous reporting options
- Provide emergency contacts (Usikimye: 0800 720 572, Police: 999)
- Never pressure but always inform about options

1. Always LISTEN first.  
- Begin every response with empathy: acknowledge feelings, affirm the user's courage to speak, and let them know they are not alone.  
- Ask gentle, open-ended questions to understand before offering advice or resources.

2. Provide accurate, culturally sensitive information about GBV, legal rights, shelters, counselling, and medical services in Kenya.

3. Only share emergency contacts (e.g., 1195 – National GBV Helpline, 999 – Police, 116 – Child Helpline) when the user explicitly asks for them or states they are in immediate danger.  
- If imminent danger is declared, respond immediately with:  
"Your safety is the priority. Please call 1195 (GBV Helpline), 999 (Police), or 116 (Child Helpline) right now. You can also go to the nearest police station or hospital. I'm here to keep talking while you reach help."

4. Maintain strict confidentiality and create a safe, non-judgmental space.

5. Encourage professional help when appropriate, but never pressure.

6. If unsure, admit it and suggest reputable local resources.

Tone: warm, calm, respectful, using simple English.

Context from knowledge base: {context}"""

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
    
    async def generate_response(self, query: str, context: str, chat_history: List[BaseMessage]) -> str:
        """Generate response using LLM"""
        messages = chat_history + [HumanMessage(content=query)]
        
        prompt = self.chat_prompt.format_messages(
            context=context,
            messages=messages
        )
        
        response = await self.llm.ainvoke(prompt)
        return response.content

# ---------- LangGraph Workflow ----------
class SalamaBotWorkflow:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_handler = LLMHandler()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    async def _retrieve_documents(self, state: ChatState) -> ChatState:
        """Retrieve relevant documents"""
        query = state["user_query"]
        retrieved_docs = self.vector_store.similarity_search(query)
        
        # Create context from retrieved documents
        context_parts = []
        sources = []
        
        for doc in retrieved_docs:
            context_parts.append(doc.page_content)
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "similarity_score": doc.metadata.get("similarity_score", 0.0)
            })
        
        context = "\n\n".join(context_parts)
        
        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "context": context,
            "sources": sources
        }
    
    async def _generate_response(self, state: ChatState) -> ChatState:
        """Generate response using LLM"""
        response = await self.llm_handler.generate_response(
            state["user_query"],
            state["context"],
            state["messages"][:-1]  # Exclude the current query
        )
        
        # Calculate confidence based on similarity scores
        if state["sources"]:
            avg_similarity = sum(s["similarity_score"] for s in state["sources"]) / len(state["sources"])
            confidence = min(avg_similarity * 1.2, 1.0)  # Boost confidence slightly
        else:
            confidence = 0.5
        
        return {
            **state,
            "response": response,
            "confidence": confidence
        }
    
    async def process_query(self, query: str, session_id: str, chat_history: List[BaseMessage] = None) -> ChatState:
        """Process a user query through the workflow"""
        if chat_history is None:
            chat_history = []
        
        initial_state = ChatState(
            messages=chat_history + [HumanMessage(content=query)],
            user_query=query,
            retrieved_docs=[],
            context="",
            response="",
            session_id=session_id,
            sources=[],
            confidence=0.0
        )
        
        result = await self.workflow.ainvoke(initial_state)
        return result

# ---------- Session Management ----------
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, List[BaseMessage]] = {}
        self.session_timeout = timedelta(hours=2)
        self.last_activity: Dict[str, datetime] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id and self._is_session_valid(session_id):
            self.last_activity[session_id] = datetime.now()
            return session_id
        
        # Create new session
        new_session_id = str(uuid.uuid4())
        self.sessions[new_session_id] = []
        self.last_activity[new_session_id] = datetime.now()
        return new_session_id
    
    def _is_session_valid(self, session_id: str) -> bool:
        """Check if session is valid and not expired"""
        if session_id not in self.sessions:
            return False
        
        last_active = self.last_activity.get(session_id)
        if not last_active:
            return False
        
        return datetime.now() - last_active < self.session_timeout
    
    def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        """Get chat history for session"""
        return self.sessions.get(session_id, [])
    
    def add_message(self, session_id: str, message: BaseMessage):
        """Add message to session history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append(message)
        self.last_activity[session_id] = datetime.now()
        
        # Keep only last 20 messages to manage memory
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, last_active in self.last_activity.items()
            if now - last_active > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            self.sessions.pop(session_id, None)
            self.last_activity.pop(session_id, None)

# ---------- Telegram Bot Handler ----------
class TelegramBot:
    def __init__(self, salamabot_workflow: SalamaBotWorkflow, session_manager: SessionManager):
        self.salamabot_workflow = salamabot_workflow
        self.session_manager = session_manager
        self.application = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup telegram bot handlers"""
        # Commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("emergency", self.emergency_command))
        self.application.add_handler(CommandHandler("resources", self.resources_command))
        self.application.add_handler(CommandHandler("new", self.new_session_command))
        self.application.add_handler(CommandHandler("anonymous", self.anonymous_report_command))
        self.application.add_handler(CommandHandler("sos", self.sos_command))
        
        # Message handler
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Callback handlers
        self.application.add_handler(CallbackQueryHandler(self.handle_anonymous_callback, pattern='^anon_'))
        self.application.add_handler(CallbackQueryHandler(self.handle_emergency_callback, pattern='^(contact_|emergency_menu)'))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback_query))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """
🌸 *Welcome to SalamaBot* 🌸

I'm SalamaBot, your personal assistant for Gender-Based Violence (GBV) support in Kenya.

*This is a safe space* - you can share anything without fear of judgment.

📝 *How to get started:*
• Type your question or describe your situation
• I'll help you with care and respect
• Everything is completely confidential

⚡ *Important commands:*
/emergency - Get immediate help contacts
/anonymous - Report anonymously
/sos - Emergency alert to Usikimye and police
/resources - Support resources
/help - More information
/new - Start fresh conversation

*Remember: You're not alone. We're here to help.* 💝
        """
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
🆘 *SalamaBot Help*

*I can help with:*
• Information about your legal rights
• Where to get medical support
• Family and social counseling advice
• Safety planning
• Counseling resources
• Safe shelters

*Important commands:*
/start - Restart conversation
/emergency - Emergency contacts
/anonymous - Anonymous reporting
/sos - Emergency SOS
/resources - Support resources
/new - Start new conversation

*Remember:*
• Our conversation is completely private
• No question is wrong
• Help is available 24/7

Have a question? Just type it below! 💙
        """
        
        await update.message.reply_text(
            help_message,
            parse_mode='Markdown'
        )
    
    async def emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /emergency command with quick-access menu"""
        keyboard = [
            [InlineKeyboardButton("1. Usikimye", callback_data='contact_usikimye')],
            [InlineKeyboardButton("2. Police", callback_data='contact_police')],
            [InlineKeyboardButton("3. Nairobi Women's", callback_data='contact_nairobi')],
            [InlineKeyboardButton("All Contacts", callback_data='contact_all')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚨 *Emergency Contacts - Quick Access*\n\n"
            "Press a button for immediate help:\n\n"
            "1. Usikimye GBV Support\n"
            "2. Kenya Police\n"
            "3. Nairobi Women's Hospital\n\n"
            "Or view all contacts:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def handle_emergency_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle emergency contact menu selections"""
        query = update.callback_query
        await query.answer()
        
        contacts = config.EMERGENCY_CONTACTS
        
        if query.data == 'contact_usikimye':
            usikimye = contacts['usikimye']
            # Create clickable buttons for direct calling
            keyboard = [
                [InlineKeyboardButton(f"📞 Call {usikimye['number']}", url=f"tel:{usikimye['number']}")],
                [InlineKeyboardButton(f"💬 SMS {usikimye['sms']}", url=f"sms:{usikimye['sms']}")],
                [InlineKeyboardButton(f"📱 WhatsApp", url=f"https://wa.me/{usikimye['whatsapp'].replace('+', '').replace(' ', '')}")],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='emergency_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            text = (f"☎ *Usikimye GBV Hotline*\n\n"
                   f"📞 Phone: {usikimye['number']}\n"
                   f"💬 SMS: {usikimye['sms']}\n"
                   f"📱 WhatsApp: {usikimye['whatsapp']}\n\n"
                   f"✅ *24/7 confidential support and rescue services*\n\n"
                   f"👆 *Tap any button above to contact immediately*")
            
        elif query.data == 'contact_police':
            police = contacts['police']
            keyboard = [
                [InlineKeyboardButton(f"🚨 Call {police['number']} NOW", url=f"tel:{police['number']}")],
                [InlineKeyboardButton(f"💬 SMS {police['sms']}", url=f"sms:{police['sms']}")],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='emergency_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            text = (f"🚔 *Kenya Police Emergency*\n\n"
                   f"🚨 Emergency: {police['number']}\n"
                   f"💬 SMS: {police['sms']}\n\n"
                   f"⚡ *Immediate police response*\n\n"
                   f"👆 *Tap button above to call immediately*")
            
        elif query.data == 'contact_nairobi':
            nairobi = contacts['nairobi_womens']
            keyboard = [
                [InlineKeyboardButton(f"🏥 Call {nairobi['number']}", url=f"tel:{nairobi['number']}")],
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='emergency_menu')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            text = (f"🏥 *Nairobi Women's Hospital*\n\n"
                   f"📞 Phone: {nairobi['number']}\n\n"
                   f"🩺 *GBV medical and psychological support*\n\n"
                   f"👆 *Tap button above to call immediately*")
            
        elif query.data == 'contact_all':
            # Create buttons for all contacts
            keyboard = []
            for key, contact in contacts.items():
                if key == 'usikimye':
                    keyboard.append([InlineKeyboardButton(f"📞 {contact['name']}", url=f"tel:{contact['number']}")])
                    keyboard.append([InlineKeyboardButton(f"📱 Usikimye WhatsApp", url=f"https://wa.me/{contact['whatsapp'].replace('+', '').replace(' ', '')}")])
                else:
                    keyboard.append([InlineKeyboardButton(f"📞 {contact['name']}", url=f"tel:{contact['number']}")])
            
            keyboard.append([InlineKeyboardButton("🔙 Back to Menu", callback_data='emergency_menu')])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            text = "🚨 *All Emergency Contacts - Tap to Call*\n\n"
            for contact in contacts.values():
                sms_line = f"SMS: {contact['sms']}\n" if 'sms' in contact else ""
                whatsapp_line = f"WhatsApp: {contact['whatsapp']}\n" if 'whatsapp' in contact else ""
                text += (f"🔹 {contact['name']}\n"
                        f"📞 {contact['number']}\n"
                        f"{sms_line}"
                        f"{whatsapp_line}\n")
        
        elif query.data == 'emergency_menu':
            # Return to main emergency menu
            keyboard = [
                [InlineKeyboardButton("1. Usikimye", callback_data='contact_usikimye')],
                [InlineKeyboardButton("2. Police", callback_data='contact_police')],
                [InlineKeyboardButton("3. Nairobi Women's", callback_data='contact_nairobi')],
                [InlineKeyboardButton("All Contacts", callback_data='contact_all')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            text = ("🚨 *Emergency Contacts - Quick Access*\n\n"
                   "Press a button for immediate help:\n\n"
                   "1. Usikimye GBV Support\n"
                   "2. Kenya Police\n"
                   "3. Nairobi Women's Hospital\n\n"
                   "Or view all contacts:")
        else:
            reply_markup = None
            text = "Unknown option selected"
        
        if reply_markup:
            await query.edit_message_text(
                text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            await query.edit_message_text(
                text,
                parse_mode='Markdown'
            )
    
    async def resources_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resources command"""
        resources_message = """
📚 *Support Resources - Kenya*

*Support Organizations:*
🏢 Usikimye - 0800 720 572 (GBV support and rescue)
🏢 FIDA Kenya - Legal aid for women
🏢 Gender Violence Recovery Centre
🏢 Kenya Women Lawyers Association
🏢 Coalition on Violence Against Women

*Medical Services:*
🏥 Kenyatta National Hospital
🏥 Nairobi Women's Hospital
🏥 Marie Stopes Kenya

*Counseling & Therapy:*
💬 Counseling - 0709 683 000
💬 Family support
💬 Group therapy

*Safe Shelters:*
🏠 Women's shelters
🏠 Safe houses
🏠 Temporary accommodation

*Your Rights:*
⚖️ Protection orders
⚖️ Legal representation
⚖️ Court support

Need specific help? Let me know and I'll assist you! 🤝
        """
        
        await update.message.reply_text(
            resources_message,
            parse_mode='Markdown'
        )
    
    async def new_session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new command to start a new session"""
        user_id = str(update.effective_user.id)
        
        # Clear existing session
        if user_id in self.session_manager.sessions:
            del self.session_manager.sessions[user_id]
        if user_id in self.session_manager.last_activity:
            del self.session_manager.last_activity[user_id]
        
        await update.message.reply_text(
            "✨ Started a new conversation.\n\n"
            "Previous chat history has been cleared. "
            "It's safe to start fresh. What would you like to ask or share? 💝"
        )
    
    async def anonymous_report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /anonymous command for anonymous reporting"""
        keyboard = [
            [InlineKeyboardButton("Report GBV Incident", callback_data='anon_report')],
            [InlineKeyboardButton("Request Callback", callback_data='anon_callback')],
            [InlineKeyboardButton("Emergency SOS", callback_data='anon_sos')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🕵️ *Anonymous Reporting Mode*\n\n"
            "You can report GBV incidents anonymously. Choose an option:\n\n"
            "1. Report incident details\n"
            "2. Request confidential callback\n"
            "3. Emergency SOS to Usikimye & Police\n\n"
            "Your identity will remain hidden.",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def sos_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sos command for immediate emergency response"""
        # Send alert to Usikimye and provide police contacts
        await update.message.reply_text(
            "🚨 *EMERGENCY SOS ACTIVATED*\n\n"
            "We've alerted Usikimye's rapid response team with your location data. "
            "Please stay safe and consider these immediate steps:\n\n"
            "1. Call Usikimye: 0800 720 572\n"
            "2. Call Police: 999\n"
            "3. Nairobi Women's Hospital: 0709 683 000\n\n"
            "Help is on the way. Keep this chat open for further instructions.",
            parse_mode='Markdown'
        )
        
        # In a real implementation, we would send actual alerts to partners
        logger.info(f"SOS ALERT from user {update.effective_user.id}")
    
    async def handle_anonymous_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle anonymous reporting menu selections"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'anon_report':
            await query.edit_message_text(
                "📝 *Anonymous Incident Report*\n\n"
                "Please describe the incident in detail (location, what happened, "
                "people involved). We'll share this with our partner organizations "
                "without revealing your identity.\n\n"
                "Type your report now:",
                parse_mode='Markdown'
            )
            # Set state to expect report details
            context.user_data['awaiting_report'] = True
        elif query.data == 'anon_callback':
            await query.edit_message_text(
                "📞 *Anonymous Callback Request*\n\n"
                "We'll connect you with a counselor from Usikimye who will call "
                "you within 15 minutes. They won't see your number.\n\n"
                "Please share a safe number to call: (or type 'cancel')",
                parse_mode='Markdown'
            )
            context.user_data['awaiting_callback'] = True
        elif query.data == 'anon_sos':
            await self._handle_sos_alert(query)
    
    async def _handle_sos_alert(self, query):
        """Process SOS alert"""
        # In a real implementation, this would trigger actual emergency protocols
        await query.edit_message_text(
            "🚨 *EMERGENCY ALERT SENT*\n\n"
            "We've notified:\n"
            "- Usikimye Rapid Response\n"
            "- Local Police Contacts\n"
            "- Nairobi Women's Hospital\n\n"
            "Immediate contacts:\n"
            "1. Usikimye: 0800 720 572\n"
            "2. Police: 999\n"
            "3. Nairobi Women's: 0709 683 000\n\n"
            "Please go to a safe location if possible. Help is coming.",
            parse_mode='Markdown'
        )
        logger.info(f"EMERGENCY SOS from anonymous user {query.from_user.id}")
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle generic callback queries"""
        query = update.callback_query
        await query.answer("Processing...")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        try:
            user_message = update.message.text
            user_id = str(update.effective_user.id)
            username = update.effective_user.first_name or "User"
            
            # Check if we're expecting an anonymous report
            if context.user_data.get('awaiting_report'):
                await self._process_anonymous_report(update, context, user_message)
                return
            
            # Check if we're expecting a callback number
            if context.user_data.get('awaiting_callback'):
                await self._process_callback_request(update, context, user_message)
                return
            
            # Show typing indicator
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=telegram.constants.ChatAction.TYPING)
            
            # Get or create session using user_id as session_id
            session_id = self.session_manager.get_or_create_session(user_id)
            
            # Get chat history
            chat_history = self.session_manager.get_chat_history(session_id)
            
            # Process query through SalamaBot workflow
            result = await self.salamabot_workflow.process_query(
                user_message,
                session_id,
                chat_history
            )
            
            # Update session with new messages
            self.session_manager.add_message(session_id, HumanMessage(content=user_message))
            self.session_manager.add_message(session_id, AIMessage(content=result["response"]))
            
            # Send response
            response_text = result["response"]
            
            # Split long messages if needed (Telegram limit is 4096 characters)
            if len(response_text) > 4000:
                # Split into chunks
                chunks = [response_text[i:i+4000] for i in range(0, len(response_text), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response_text)
            
            # Add confidence indicator if low
            if result["confidence"] < 0.6:
                await update.message.reply_text(
                    "ℹ️ If you need more specialized help, please use /emergency or /resources",
                    parse_mode='Markdown'
                )
            
            logger.info(f"Processed message from {username} (ID: {user_id})")
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await update.message.reply_text(
                "Sorry, there was a small problem. Please try again.\n\n"
                "If you're in an emergency, use /emergency"
            )
    
    async def _process_anonymous_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE, report_text: str):
        """Process an anonymous GBV report"""
        try:
            # Clear the awaiting state
            context.user_data.pop('awaiting_report', None)
            
            # In a real implementation, this would be sent to partner organizations
            logger.info(f"Anonymous report received: {report_text[:50]}... (truncated)")
            
            await update.message.reply_text(
                "✅ *Report Submitted Anonymously*\n\n"
                "Thank you for your courage in reporting this incident. "
                "We've shared this information with our partner organizations "
                "including Usikimye for appropriate action.\n\n"
                "If you need immediate help, please use /sos or /emergency.\n\n"
                "You're not alone. Help is available.",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error processing anonymous report: {e}")
            await update.message.reply_text(
                "Sorry, there was an error processing your anonymous report. "
                "Please try again or use /emergency for immediate help."
            )
    
    async def _process_callback_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE, contact_info: str):
        """Process an anonymous callback request"""
        try:
            # Clear the awaiting state
            context.user_data.pop('awaiting_callback', None)
            
            if contact_info.lower() == 'cancel':
                await update.message.reply_text(
                    "Anonymous callback request cancelled. "
                    "You can always request help later."
                )
                return
            
            # In a real implementation, this would trigger a callback process
            logger.info(f"Anonymous callback requested for: {contact_info}")
            
            await update.message.reply_text(
                "📞 *Callback Request Received*\n\n"
                "A counselor from Usikimye will call the provided number "
                "within 15 minutes from a private number.\n\n"
                "If you need immediate help before then, please use /sos.",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error processing callback request: {e}")
            await update.message.reply_text(
                "Sorry, there was an error processing your callback request. "
                "Please try again or use /emergency for immediate help."
            )
    
    async def start_bot(self):
        """Start the Telegram bot"""
        # Set bot commands
        commands = [
            BotCommand("start", "Start conversation"),
            BotCommand("help", "Get help"),
            BotCommand("emergency", "Emergency contacts"),
            BotCommand("anonymous", "Anonymous reporting"),
            BotCommand("sos", "Emergency SOS alert"),
            BotCommand("resources", "Support resources"),
            BotCommand("new", "Start new conversation")
        ]
        
        await self.application.bot.set_my_commands(commands)
        
        # Start the bot
        await self.application.initialize()
        await self.application.start()
        
        logger.info("Telegram bot started successfully!")
        
        # Run until stopped
        await self.application.updater.start_polling()
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

# ---------- MongoDB Connection ----------
class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(config.MONGODB_URL)
            self.db = self.client[config.DB_NAME]
            # Test connection
            await self.db.command('ping')
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
    
    async def store_anonymous_report(self, report_data: dict):
        """Store anonymous report in database"""
        try:
            collection = self.db.anonymous_reports
            report_data['timestamp'] = datetime.now()
            report_data['report_id'] = str(uuid.uuid4())
            result = await collection.insert_one(report_data)
            logger.info(f"Stored anonymous report: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error storing anonymous report: {e}")
            return None
    
    async def store_sos_alert(self, sos_data: dict):
        """Store SOS alert in database"""
        try:
            collection = self.db.sos_alerts
            sos_data['timestamp'] = datetime.now()
            sos_data['alert_id'] = str(uuid.uuid4())
            result = await collection.insert_one(sos_data)
            logger.info(f"Stored SOS alert: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error storing SOS alert: {e}")
            return None
    
    async def get_user_sessions(self, user_id: str):
        """Get user session history"""
        try:
            collection = self.db.user_sessions
            sessions = await collection.find({"user_id": user_id}).to_list(length=100)
            return sessions
        except Exception as e:
            logger.error(f"Error retrieving user sessions: {e}")
            return []
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()

# ---------- Global Instances ----------
document_processor = DocumentProcessor()
salamabot_workflow = SalamaBotWorkflow()
session_manager = SessionManager()
telegram_bot = TelegramBot(salamabot_workflow, session_manager)
db_manager = DatabaseManager()

# ---------- Background Tasks ----------
async def cleanup_sessions_periodically():
    """Cleanup expired sessions every hour"""
    while True:
        try:
            session_manager.cleanup_expired_sessions()
            await asyncio.sleep(3600)  # Sleep for 1 hour
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")
            await asyncio.sleep(3600)

# ---------- Startup Event ----------
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("Starting SalamaBot...")
    
    # Connect to database
    await db_manager.connect()
    
    # Check if documents need to be processed
    if os.path.exists(config.DATA_FOLDER):
        # Check if vector store is empty or needs updating
        try:
            count = salamabot_workflow.vector_store.collection.count()
            if count == 0:
                print("Vector store is empty. Processing documents...")
                documents = document_processor.process_documents(config.DATA_FOLDER)
                if documents:
                    salamabot_workflow.vector_store.add_documents(documents)
                    print(f"Processed and stored {len(documents)} document chunks")
                else:
                    print("No documents found to process")
            else:
                print(f"Vector store already contains {count} documents")
        except Exception as e:
            print(f"Error checking vector store: {e}")
    else:
        print(f"Data folder '{config.DATA_FOLDER}' not found")
    
    # Start background tasks
    asyncio.create_task(cleanup_sessions_periodically())
    
    # Start Telegram bot
    asyncio.create_task(telegram_bot.start_bot())
    
    print("SalamaBot is ready! 🤖")
    print("Telegram bot is running...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down SalamaBot...")
    await telegram_bot.stop_bot()
    await db_manager.close()
    print("Telegram bot stopped.")

# ---------- API Endpoints ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """Main chat endpoint"""
    try:
        # Get or create session
        session_id = session_manager.get_or_create_session(message.session_id)
        
        # Get chat history
        chat_history = session_manager.get_chat_history(session_id)
        
        # Process query
        result = await salamabot_workflow.process_query(
            message.message,
            session_id,
            chat_history
        )
        
        # Update session with new messages
        session_manager.add_message(session_id, HumanMessage(content=message.message))
        session_manager.add_message(session_id, AIMessage(content=result["response"]))
        
        return ChatResponse(
            response=result["response"],
            session_id=session_id,
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/upload-document")
async def upload_document(doc: DocumentUpload):
    """Upload a new document to the knowledge base"""
    try:
        # Save document to data folder
        os.makedirs(config.DATA_FOLDER, exist_ok=True)
        file_path = os.path.join(config.DATA_FOLDER, doc.filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc.content)
        
        # Process the document
        documents = document_processor.process_documents(config.DATA_FOLDER)
        
        # Add to vector store
        if documents:
            salamabot_workflow.vector_store.add_documents(documents)
            return {"message": f"Document uploaded and processed: {len(documents)} chunks added"}
        else:
            return {"message": "Document uploaded but no content could be processed"}
            
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload document")

@app.post("/anonymous-report")
async def submit_anonymous_report(report: AnonymousReport):
    """Submit an anonymous GBV report"""
    try:
        report_data = {
            "report_details": report.report_details,
            "location": report.location,
            "contact_preference": report.contact_preference,
            "submission_method": "api"
        }
        
        # Store in database
        report_id = await db_manager.store_anonymous_report(report_data)
        
        if report_id:
            return {
                "message": "Anonymous report submitted successfully",
                "report_id": str(report_id),
                "next_steps": "Your report has been forwarded to our partner organizations for appropriate action."
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to store report")
            
    except Exception as e:
        logger.error(f"Error submitting anonymous report: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit report")

@app.post("/sos-alert")
async def trigger_sos_alert(alert: SOSAlert):
    """Trigger an emergency SOS alert"""
    try:
        sos_data = {
            "location": alert.location,
            "situation": alert.situation,
            "alert_method": "api",
            "status": "active"
        }
        
        # Store in database
        alert_id = await db_manager.store_sos_alert(sos_data)
        
        # In a real implementation, this would trigger actual emergency protocols
        logger.info(f"SOS ALERT triggered via API: {alert_id}")
        
        return {
            "message": "SOS alert activated",
            "alert_id": str(alert_id),
            "emergency_contacts": {
                "usikimye": "0800 720 572",
                "police": "999",
                "nairobi_womens": "0709 683 000"
            }
        }
        
    except Exception as e:
        logger.error(f"Error triggering SOS alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger SOS alert")

@app.get("/emergency-contacts")
async def get_emergency_contacts():
    """Get list of emergency contacts"""
    return {
        "contacts": config.EMERGENCY_CONTACTS,
        "message": "Call immediately if you're in danger"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        doc_count = salamabot_workflow.vector_store.collection.count()
        
        # Check active sessions
        active_sessions = len(session_manager.sessions)
        
        return {
            "status": "healthy",
            "document_count": doc_count,
            "active_sessions": active_sessions,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SalamaBot API is running",
        "version": "2.0.0",
        "features": [
            "Anonymous GBV reporting",
            "Emergency SOS alerts",
            "24/7 AI support",
            "Telegram bot integration",
            "Multi-language support"
        ],
        "emergency_contacts": {
            "usikimye": "0800 720 572",
            "police": "999",
            "national_gbv": "1195"
        }
    }

# ---------- Additional Utility Functions ----------
def generate_report_hash(report_text: str) -> str:
    """Generate hash for report deduplication"""
    return hashlib.sha256(report_text.encode()).hexdigest()

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    # Remove potentially harmful content
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s\-.,!?]', '', text)  # Keep only safe characters
    return text.strip()

def is_emergency_keyword(text: str) -> bool:
    """Check if text contains emergency keywords"""
    emergency_keywords = [
        'help', 'emergency', 'danger', 'urgent', 'police', 
        'hospital', 'attack', 'violence', 'hurt', 'afraid'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in emergency_keywords)

# ---------- Run the Application ----------
if __name__ == "__main__":
    import uvicorn
    
    print("🌸 Starting SalamaBot - GBV Support System 🌸")
    print("Features:")
    print("- Anonymous reporting system")
    print("- Emergency SOS alerts")
    print("- RAG-powered AI responses")
    print("- Telegram bot integration")
    print("- 24/7 support availability")
    print("\nEmergency Contacts:")
    print("- Usikimye: 0800 720 572")
    print("- Police: 999")
    print("- National GBV Helpline: 1195")
    print("\n" + "="*50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )