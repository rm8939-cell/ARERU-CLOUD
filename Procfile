web: gunicorn web_app:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 90 --graceful-timeout 30 --max-requests 80 --max-requests-jitter 20 --worker-class sync
