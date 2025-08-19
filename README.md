# Clean Connection - Web Automation Agent

A powerful web automation agent built with Playwright MCP (Model Context Protocol), LangChain, and Google Gemini AI. This agent can navigate websites, fill forms, click buttons, and perform complex web automation tasks intelligently.

## 🚀 Features

- **Intelligent Web Automation**: Uses AI to understand and execute complex web tasks
- **Playwright Integration**: Built on top of Playwright for reliable browser automation
- **MCP Protocol**: Uses Model Context Protocol for tool communication
- **LangChain Integration**: Leverages LangChain for AI agent orchestration
- **Google Gemini AI**: Powered by Google's latest AI model for intelligent decision making
- **Form Filling**: Automatically detects and fills web forms with dummy data
- **Navigation**: Smart navigation with loop detection and error handling
- **Screenshot Support**: Built-in screenshot capabilities for debugging

## 🛠️ Prerequisites

- Python 3.11+
- Node.js 18+ with npx
- Google AI API key
- macOS/Linux/Windows

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd clean-connection
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Playwright MCP
```bash
npx -y @playwright/mcp@latest --help
```

### 5. Configure API Key
```bash
cp config.template.json config.json
# Edit config.json and add your Google API key
```

## 🔑 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google AI API key
- `MODEL_NAME`: AI model to use (default: gemini-2.5-flash-lite)

### Browser Options
The agent runs in headless mode by default. You can modify `main.py` to change browser settings:
- `--headless`: Run browser in background
- `--isolated`: Use isolated browser profile
- `--browser`: Choose browser (chrome, firefox, webkit)

## 🚀 Usage

### Basic Usage
```bash
python3 main.py
```

### Custom Goal
```bash
python3 main.py "Navigate to https://example.com and click the login button"
```

### With Custom Configuration
```bash
python3 main.py --api-key YOUR_KEY --model gemini-2.5-flash-lite --max-steps 100
```

## 🧪 Testing

Run the built-in tests to verify everything works:
```bash
python3 tool_manager.py --run-tests
```

Or run individual tests:
```bash
python3 tool_manager.py --single-test nav --url https://example.com
```

## 📁 Project Structure

```
clean-connection/
├── main.py              # Main entry point
├── agent_core.py        # Core agent logic and workflow
├── tool_manager.py      # MCP tool management and execution
├── utils.py             # Utility functions and helpers
├── prompts.py           # AI prompt templates
├── requirements.txt     # Python dependencies
├── config.template.json # Configuration template
├── README.md           # This file
└── venv/               # Virtual environment (not in git)
```

## 🔧 How It Works

1. **Initialization**: Connects to Playwright MCP server
2. **Tool Discovery**: Automatically discovers available browser automation tools
3. **Goal Analysis**: AI analyzes the user's goal and creates a plan
4. **Execution**: Agent executes tools in sequence to complete the task
5. **Reflection**: Agent reflects on results and adjusts strategy
6. **Completion**: Task is completed or maximum steps reached

## 🛡️ Security

- API keys are stored in `config.json` (not committed to git)
- Browser runs in isolated mode by default
- Headless mode prevents unauthorized access
- All sensitive data is excluded via `.gitignore`

## 🐛 Troubleshooting

### Common Issues

1. **"Browser already in use"**: Kill existing Playwright processes
   ```bash
   pkill -f "playwright"
   ```

2. **Recursion limit reached**: Increase `recursion_limit` in main.py

3. **MCP connection failed**: Ensure Node.js and npx are installed
   ```bash
   which npx && npx --version
   ```

4. **API key errors**: Verify your Google API key in `config.json`

### Debug Mode
Run with verbose logging:
```bash
python3 main.py --debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Playwright](https://playwright.dev/) for browser automation
- [LangChain](https://langchain.com/) for AI agent framework
- [Google Gemini](https://ai.google.dev/) for AI capabilities
- [MCP](https://modelcontextprotocol.io/) for tool communication

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Note**: This is a web automation tool. Please use responsibly and in accordance with website terms of service.
