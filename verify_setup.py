"""Quick sanity check to verify Gemini API connection and environment setup."""

import os
import sys
from dotenv import load_dotenv

# Ensure Windows terminal handles UTF-8 characters safely
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 1. Load variables from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_api_key_here":
    print("[ERROR] GEMINI_API_KEY not found or not set in .env")
    print("Please create a .env file and add your actual API key:")
    print("   GEMINI_API_KEY=\"AIzaSy...\"")
    sys.exit(1)

# Mask key for privacy
masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
print(f"[OK] Found GEMINI_API_KEY: {masked}")

# 2. Test Gemini API connection
try:
    from google import genai
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'Connection successful!'"
    )
    print(f"[OK] Gemini Response: {response.text.strip()}")
    print("\n[SUCCESS] Environment and API key verification passed! Ready for Step 2.")
except Exception as e:
    print(f"[ERROR] Failed to connect to Gemini API: {e}")
    sys.exit(1)
