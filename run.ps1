$env:PYTHONPATH = (Get-Location).Path
Start-Process -FilePath "C:/Users/lucka/AppData/Local/Programs/Python/Python312/python.exe" -ArgumentList "-m","uvicorn","app:app","--host","0.0.0.0","--port","8000" -NoNewWindow
Start-Process -FilePath "C:/Users/lucka/AppData/Local/Programs/Python/Python312/python.exe" -ArgumentList "-m","streamlit","run","ui/dashboard.py","--server.port","8501" -NoNewWindow
