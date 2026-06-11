FROM python:3.10-slim

WORKDIR /app

# Install poetry
RUN pip install poetry==1.6.1

# Copy project files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --only main

# Copy application code
COPY api/ ./api/
COPY core/ ./core/
COPY ai/ ./ai/
COPY db/ ./db/

# Expose API port
EXPOSE 3000

# Start FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000"]
