FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories for data
RUN mkdir -p data/uploads data/vectorstore

EXPOSE 8501

CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]