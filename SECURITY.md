# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please open an issue or contact the maintainers directly.

## Known Limitations (for production deployment)

### 1. CORS Configuration
The default CORS configuration allows all origins (`*`), which is appropriate for local development but **not for production**.

**For production**, modify `backend/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Restrict to your domain
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    allow_credentials=False,
)
```

### 2. Authentication
This application has **no built-in authentication**. It's designed for local-first usage where data never leaves your machine.

**For production deployments**, add authentication middleware:
- API key authentication
- OAuth2 / JWT tokens
- Session-based auth

### 3. Rate Limiting
No rate limiting is implemented. For public-facing deployments, add rate limiting to prevent DoS attacks:
- Use `slowapi` or similar FastAPI rate limiting middleware
- Configure limits per endpoint

### 4. Session Storage
Sessions are stored in memory and lost on server restart. For production:
- Use Redis or a database for session persistence
- Implement session expiration/cleanup

### 5. File Upload Security
The following protections are implemented:
- **Path traversal prevention**: Filenames are sanitized before use
- **Extension whitelist**: Only allowed file types are accepted
- **Size limit**: 500MB max upload

### 6. SQL Injection Prevention
All DuckDB queries use sanitized paths with escaped single quotes. The `_safe_path()` function handles this across all modules.

## Best Practices for Deployment

1. **Run behind a reverse proxy** (nginx, Caddy, Traefik)
2. **Use HTTPS** with valid certificates
3. **Restrict CORS** to your actual domain
4. **Add authentication** if exposing publicly
5. **Set appropriate rate limits**
6. **Log all requests** for audit purposes
7. **Run with minimal permissions** (non-root user)

## Data Privacy

PreTrainAudit is designed as a **local-first** application:
- All processing happens on your machine
- Data is never sent to external servers
- Uploaded files are stored locally in `backend/uploads/`
- Delete sessions via the `/session/{session_id}` endpoint to clean up files

## Security Fixes Applied

| Issue | Severity | Fix |
|-------|----------|-----|
| SQL injection via path interpolation | CRITICAL | Added `_safe_path()` with quote escaping |
| Path traversal in upload | HIGH | Added `_sanitize_filename()` and extension whitelist |
