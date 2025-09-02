FROM python:3.10-slim

WORKDIR /app

COPY setup.cfg pyproject.toml requirements.txt requirements-dev.txt ./
RUN pip install -e .[dev]

COPY . .

CMD ["pytest"]