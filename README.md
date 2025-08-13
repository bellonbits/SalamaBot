# SalamaBot

**Enhanced GBV Support System for Kenya with Anonymous Reporting and Emergency SOS Features**

SalamaBot is a compassionate AI-powered chatbot designed to provide immediate support, information, and emergency assistance for Gender-Based Violence (GBV) survivors in Kenya. The system combines advanced AI technology with human-centered design to create a safe, confidential, and accessible support platform.

## Key Features

### AI-Powered Support
- **RAG (Retrieval-Augmented Generation)** system for accurate, contextual responses
- **Multi-language support** with culturally sensitive interactions
- **24/7 availability** for immediate assistance
- **Session management** for continuous conversation context

### Anonymous Reporting System
- **Completely anonymous** incident reporting
- **Secure data handling** with encryption
- **Partner organization integration** for appropriate action
- **Callback request system** for confidential counselor contact

###  Emergency SOS Features
- **One-tap emergency alerts** to Usikimye and police
- **Immediate contact access** with clickable phone numbers
- **Location-based assistance** for rapid response
- **Emergency contact directory** with multiple support options

### Telegram Bot Integration
- **Intuitive command system** for easy navigation
- **Inline keyboards** for quick access to emergency contacts
- **Message splitting** for long responses
- **Typing indicators** for better user experience

## Architecture

### Core Components

1. **FastAPI Backend** - RESTful API with async support
2. **LangGraph Workflow** - AI conversation management
3. **Vector Database (ChromaDB)** - Document retrieval system
4. **MongoDB** - Session and report storage
5. **Telegram Bot** - User interface
6. **Document Processor** - PDF text extraction and chunking

### Technology Stack

- **Python 3.8+**
- **FastAPI** - Web framework
- **LangChain & LangGraph** - AI workflow management
- **Groq LLM** - Language model (Llama-4-Scout-17B)
- **ChromaDB** - Vector database
- **MongoDB** - Document database
- **Telegram Bot API** - Messaging platform
- **Sentence Transformers** - Text embeddings

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- MongoDB instance
- Telegram Bot Token
- Groq API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SalamaBot-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create a .env file or set environment variables
   export GROQ_API_KEY="your_groq_api_key"
   export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
   export MONGODB_URL="mongodb://localhost:27017"
   ```

4. **Prepare data folder**
   ```bash
   mkdir data
   # Add PDF documents to the data folder for knowledge base
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

The application will start on `http://localhost:8000` with the Telegram bot running simultaneously.

## 📚 API Documentation

### Core Endpoints

#### Chat Interface
```http
POST /chat
Content-Type: application/json

{
  "message": "I need help with a situation",
  "session_id": "optional-session-id",
  "user_id": "optional-user-id"
}
```

#### Anonymous Reporting
```http
POST /anonymous-report
Content-Type: application/json

{
  "report_details": "Detailed description of the incident",
  "location": "Nairobi, Kenya",
  "contact_preference": "phone"
}
```

#### Emergency SOS Alert
```http
POST /sos-alert
Content-Type: application/json

{
  "location": "Current location",
  "situation": "Emergency description"
}
```

#### Document Upload
```http
POST /upload-document
Content-Type: application/json

{
  "filename": "gbv_guidelines.pdf",
  "content": "Document content as text"
}
```

### Health & Monitoring
```http
GET /health
GET /emergency-contacts
GET /
```

## 🤖 Telegram Bot Commands

### Core Commands
- `/start` - Initialize conversation and show welcome message
- `/help` - Display help information and available commands
- `/new` - Start a fresh conversation session
- `/emergency` - Access emergency contacts with quick-call buttons
- `/anonymous` - Enter anonymous reporting mode
- `/sos` - Trigger emergency SOS alert
- `/resources` - View support resources and organizations

### Emergency Contact Quick Access
The bot provides one-tap access to:
- **Usikimye GBV Hotline**: 0800 720 572
- **Kenya Police**: 999
- **Nairobi Women's Hospital**: 0709 683 000
- **National GBV Helpline**: 1195
- **Childline Kenya**: 116

## 🔒 Security & Privacy

### Data Protection
- **End-to-end encryption** for all communications
- **Anonymous reporting** with no user identification
- **Session timeout** (2 hours) for automatic cleanup
- **Input sanitization** to prevent malicious content
- **Secure API endpoints** with proper authentication

### Confidentiality Features
- **No user data logging** in production
- **Anonymous session management**
- **Secure document storage**
- **Partner organization integration** without user identification

## 🏥 Emergency Support Network

### Partner Organizations
- **Usikimye** - GBV support and rescue services
- **Kenya Police** - Emergency law enforcement response
- **Nairobi Women's Hospital** - Medical and psychological support
- **National GBV Helpline** - 24/7 support hotline
- **Childline Kenya** - Child protection services

### Response Protocols
1. **Immediate Assessment** - AI evaluates urgency level
2. **Emergency Routing** - Direct connection to appropriate services
3. **Follow-up Support** - Continuous assistance and guidance
4. **Resource Provision** - Access to legal, medical, and counseling services

## 🛠️ Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
MONGODB_URL=mongodb://localhost:27017
DATA_FOLDER=data
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=salamabot_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

### Customization Options
- **Emergency Contacts** - Modify contact information in `Config.EMERGENCY_CONTACTS`
- **AI Model** - Change LLM provider or model in configuration
- **Response Templates** - Customize system prompts and responses
- **Session Timeout** - Adjust session duration in `SessionManager`

## 📊 Monitoring & Analytics

### Health Checks
- **Vector store status** - Document count and availability
- **Active sessions** - Current user sessions
- **Database connectivity** - MongoDB connection status
- **API response times** - Performance monitoring

### Logging
- **Structured logging** with different levels
- **Error tracking** for debugging
- **Usage analytics** (anonymous)
- **Emergency alert logging**

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests and linting
5. Submit a pull request

### Code Standards
- **Type hints** for all functions
- **Docstrings** for classes and methods
- **Error handling** with proper logging
- **Security best practices** implementation

##  License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

##  Emergency Information

### Immediate Help
If you're in immediate danger:
- **Call Usikimye**: 0800 720 572
- **Call Police**: 999
- **National GBV Helpline**: 1195

### Support Resources
- **Legal Aid**: FIDA Kenya
- **Medical Support**: Nairobi Women's Hospital
- **Counseling**: Various partner organizations
- **Safe Shelters**: Network of secure locations

## 📞 Contact & Support

For technical support or questions about the SalamaBot system:
- **GitHub Issues**: Report bugs or feature requests
- **Documentation**: Check this README and inline code comments
- **Emergency**: Use the built-in emergency features

---

**Remember: You're not alone. Help is available 24/7. 🌸**

*SalamaBot is designed with care and compassion to support survivors of gender-based violence in Kenya. Every interaction is confidential, respectful, and aimed at providing the support you need.*
