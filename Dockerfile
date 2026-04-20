FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src
ENV PORT=8080

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f /app/app_bundle_payload.py ]; then \
        echo 'embedded_bundle_detected: /app/app_bundle_payload.py'; \
        rm -rf /app/src /app/docs /app/scripts /app/data/import_templates && \
        python /app/app_bundle_payload.py /app && \
        test -f /app/src/tw_stock_ai/adapters/__init__.py || (echo 'bundle_extract_missing: src/tw_stock_ai/adapters/__init__.py' >&2; exit 1) && \
        test -f /app/src/tw_stock_ai/routers/api.py || (echo 'bundle_extract_missing: src/tw_stock_ai/routers/api.py' >&2; exit 1) && \
        pip install --no-cache-dir -r requirements.txt; \
    elif [ -f /app/UPLOAD_MANIFEST.txt ] && [ ! -f /app/app_bundle.zip ]; then \
        echo 'archive_bundle_missing: /app/app_bundle.zip' >&2; \
        exit 1; \
    elif [ -f /app/app_bundle.zip ]; then \
        echo 'archive_bundle_detected: /app/app_bundle.zip'; \
        rm -rf /app/src /app/docs /app/scripts /app/data/import_templates && \
        python -c "import zipfile; zipfile.ZipFile('/app/app_bundle.zip').extractall('/app')" && \
        test -f /app/src/tw_stock_ai/adapters/__init__.py || (echo 'bundle_extract_missing: src/tw_stock_ai/adapters/__init__.py' >&2; exit 1) && \
        test -f /app/src/tw_stock_ai/routers/api.py || (echo 'bundle_extract_missing: src/tw_stock_ai/routers/api.py' >&2; exit 1) && \
        pip install --no-cache-dir -r requirements.txt; \
    elif ls /app/*.whl >/dev/null 2>&1; then \
        pip install --no-cache-dir /app/*.whl; \
    else \
        test -f /app/requirements.txt || (echo "requirements.txt missing and no wheel provided" >&2; exit 1); \
        test -f /app/pyproject.toml || (echo "pyproject.toml missing and no wheel provided" >&2; exit 1); \
        test -d /app/src || (echo "src directory missing and no wheel provided" >&2; exit 1); \
        pip install --no-cache-dir -r requirements.txt && \
        pip install --no-cache-dir -e .; \
    fi && \
    python -c "import tw_stock_ai; import tw_stock_ai.main; import tw_stock_ai.adapters; import tw_stock_ai.worker"

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn tw_stock_ai.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
