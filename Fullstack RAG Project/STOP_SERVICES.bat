@echo off
title Stop RAG Services
echo Stopping OpenSearch Docker containers...
docker compose down
echo [OK] Services stopped.
pause
