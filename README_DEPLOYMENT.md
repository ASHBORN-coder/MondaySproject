# 🚀 Streamlit Cloud Deployment Guide

## 📁 This Folder Contains Everything You Need

This `STREAMLIT_DEPLOY` folder has all the files required to deploy your Monday.com BI Agent to Streamlit Cloud.

---

## 📋 Files Included

```
STREAMLIT_DEPLOY/
├── MainScript_Simple.py          # Main app (entry point)
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Excludes sensitive files
├── src/                          # Core modules
│   ├── __init__.py
│   ├── schema_manager.py         # Monday.com API integration
│   ├── llm_query_analyzer.py     # Query analysis (Gemini 1.5 Flash)
│   └── llm_code_executor.py      # Code generation & execution
└── .streamlit/
    └── config.toml              # UI configuration
```

---

## 🚀 Deployment Steps

### Step 1: Push to GitHub

```bash
cd STREAMLIT_DEPLOY
git init
git add .
git commit -m "Monday.com BI Agent - Ready for deployment"
git remote add origin https://github.com/YOUR_USERNAME/monday-bi-agent.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Click **"New app"**
3. Connect your GitHub repository
4. Configure:
   - **Repository:** YOUR_USERNAME/monday-bi-agent
   - **Branch:** main
   - **Main file path:** `MainScript_Simple.py`
   - **Python version:** 3.11 or higher

### Step 3: Add Secrets

In Streamlit Cloud dashboard → **App settings** → **Secrets**:

```toml
MONDAY_API_TOKEN = "your_monday_api_token"
GOOGLE_API_KEY = "your_google_gemini_api_key"
MONDAY_DEALS_BOARD_ID = "your_deals_board_id"
MONDAY_ORDERS_BOARD_ID = "your_orders_board_id"
```

**Important:** Use your actual API keys and board IDs!

---

## ✅ Pre-Deployment Checklist

- [x] All core files included
- [x] requirements.txt with dependencies
- [x] .gitignore excludes sensitive files
- [x] LLM model set to gemini-1.5-flash (free tier)
- [x] Unified polishing strategy implemented
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Configure secrets

---

## 🔐 Security Notes

1. **Never commit API keys** - Use Streamlit Cloud secrets
2. **.gitignore** already excludes `.env` and sensitive files
3. **Secrets are encrypted** in Streamlit Cloud

---

## 🧪 Test Locally First

Before deploying, test locally:

```bash
cd STREAMLIT_DEPLOY
pip install -r requirements.txt
streamlit run MainScript_Simple.py
```

If it works locally, it will work on Streamlit Cloud!

---

## 📊 After Deployment

Your app will be available at:
```
https://your-app-name.streamlit.app
```

Users can:
- Ask natural language questions
- Get AI-powered insights
- Analyze deals and work orders
- See revenue breakdowns by sector

---

## 🎯 Features

- **Smart Column Selection:** LLM determines which data to fetch
- **Pandas Code Generation:** Generates and executes analysis code
- **Unified Data Polishing:** Ensures consistency across boards
- **Duplicate Detection:** Warns about potential double counting
- **Status Filtering:** Intelligently filters by deal status

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **LLM:** Google Gemini 2.5 Flash (via LangChain)
- **API:** Monday.com GraphQL API

---

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify secrets are configured correctly
3. Ensure API keys have proper permissions
4. Test locally to isolate the issue

---

**Ready to deploy! 🚀**
