# A_Team_Agent Development

## Prerequisites
- Python 3.8 or higher installed on your system

## Quick Start

### Option 1: Easy Setup (Windows - Recommended for Beginners)

1. **Run the setup script:**
   - Double-click `setup.bat` 
   - This will create the virtual environment, copy `.env.example` to `.env`, and install all dependencies

2. **Edit `.env` file and add your OpenAI API key:**
   - Open `.env` file in any text editor
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

3. **Run the application:**
   - Double-click `run.bat`
   - Your browser will open automatically

### Option 2: Manual Setup (All Platforms)

1. **Create environment file:**
```bash
# Copy the example file
cp .env.example .env
```

2. **Create a virtual environment:**
```bash
python -m venv .venv
```

3. **Activate the virtual environment:**

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

4. **Edit `.env` file and add your OpenAI API key:**

Replace the placeholder in the `.env` file:
```
OPENAI_API_KEY=your_actual_api_key_here
```

5. **Install dependencies:**
```bash
pip install -r requirements.txt
```

6. **Run the application:**
```bash
streamlit run app.py
```

6. **Open your browser to `http://localhost:8501`**


## File Structure

- `setup.bat` - Windows setup script (creates .venv, copies .env, installs dependencies)
- `run.bat` - Windows run script (starts the app)
- `app.py` - Main Streamlit application
- `src/` - Core application modules
- `data/` - Document storage and vector indices
- `tests/` - Unit tests
- `.env.example` - Environment variables template (gets pushed to repo)
- `.env` - Your local environment variables (edit with your API key)
- `.venv/` - Virtual environment (created by setup.bat or manually)

## Troubleshooting

**Virtual Environment Issues:**
- Make sure you see `(.venv)` in your terminal prompt after activation
- If activation fails, try `python3` instead of `python`

**OpenAI API Issues:**
- Ensure your API key is valid and has sufficient credits
- Make sure there are no extra spaces in the `.env` file

**Vector Store Issues:**
- Delete `data/vector_store/` directory to reset the index

**File Upload Issues:**
- Supported formats: PDF, TXT, MD, DOCX

**Dependencies Issues:**
- If installation fails, try upgrading pip: `pip install --upgrade pip`