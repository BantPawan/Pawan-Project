# Set up ngrok and run Streamlit
!pip install -q pyngrok
from pyngrok import ngrok
import subprocess
import threading
import time

# Kill existing ngrok and Streamlit processes
!pkill ngrok
!pkill -f streamlit

# Set ngrok auth token
ngrok.set_auth_token("2vaEHf6f3GHr6NuvDwbixSfPqp6_5CgasErmF2bLCbPozsGzF")  # Your ngrok token

# Run Streamlit in background
def run_streamlit():
    subprocess.run(
        ["nohup", "streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        stdout=open('streamlit.log', 'w'),
        stderr=subprocess.STDOUT
    )

thread = threading.Thread(target=run_streamlit, daemon=True)
thread.start()

# Wait for Streamlit to initialize
time.sleep(8)

# Set up ngrok tunnel
try:
    public_url = ngrok.connect(8501, "http")
    print("\n⭐ Your app is now available at:", public_url)
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("\n🚫 Shutting down...")
except Exception as e:
    print(f"⚠️ Error: {str(e)}")
finally:
    ngrok.kill()
    print("✅ Ngrok tunnel closed")
    !pkill -f streamlit
    print("✅ Streamlit server stopped")
