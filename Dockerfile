FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
# Copy model artifact
COPY pet_logreg_model.pth .

# Set python path so imports work
ENV PYTHONPATH=/app/src

EXPOSE 5000

# Run Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.server:app"]
