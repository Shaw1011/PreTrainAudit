"""
main.py — PreTrainAudit API (stable build)

Opening: http://localhost:8000 serves the frontend.
API docs: http://localhost:8000/docs

FIXES applied:
  - datetime.utcnow() → datetime.now(timezone.utc)  [deprecated in Python 3.12+]
    Occurrences: _cleanup_expired_sessions() × 1, upload handler × 1
  - Import: added `timezone` to datetime imports
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, field_validator
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta, timezone   # ← added timezone
import os, uuid, logging
import asyncio
import shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pretrainaudit")

BASE_DIR     = Path(__file__).parent.resolve()
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR   = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES        = 500 * 1024 * 1024  # 500 MB
SESSION_TTL_SECONDS     = 3600               # 1 hour default TTL
CLEANUP_INTERVAL_SECONDS = 300               # Run cleanup every 5 minutes

# ── Module imports (each wrapped so one bad module doesn't kill the server) ────
def _try_import():
    mods = {}
    try:
        from modules.ingestion import ingest_dataset
        mods["ingest_dataset"] = ingest_dataset
    except Exception as e:
        logger.error(f"ingestion import failed: {e}")

    try:
        from modules.profiler import profile_dataset
        mods["profile_dataset"] = profile_dataset
    except Exception as e:
        logger.error(f"profiler import failed: {e}")

    try:
        from modules.quality import compute_quality_score
        mods["compute_quality_score"] = compute_quality_score
    except Exception as e:
        logger.error(f"quality import failed: {e}")

    try:
        from modules.adversarial import scan_adversarial_vulnerability
        mods["scan_adversarial_vulnerability"] = scan_adversarial_vulnerability
    except Exception as e:
        logger.error(f"adversarial import failed: {e}")

    try:
        from modules.memorization import compute_memorization_risk
        mods["compute_memorization_risk"] = compute_memorization_risk
    except Exception as e:
        logger.error(f"memorization import failed: {e}")

    try:
        from modules.fairness import predict_fairness_drift
        mods["predict_fairness_drift"] = predict_fairness_drift
    except Exception as e:
        logger.error(f"fairness import failed: {e}")

    try:
        from modules.contamination import detect_benchmark_contamination
        mods["detect_benchmark_contamination"] = detect_benchmark_contamination
    except Exception as e:
        logger.error(f"contamination import failed: {e}")

    try:
        from modules.cost_estimator import estimate_training_cost
        mods["estimate_training_cost"] = estimate_training_cost
    except Exception as e:
        logger.error(f"cost_estimator import failed: {e}")

    try:
        from modules.compatibility import evaluate_compatibility
        mods["evaluate_compatibility"] = evaluate_compatibility
    except Exception as e:
        logger.error(f"compatibility import failed: {e}")

    try:
        from modules.report import generate_report
        mods["generate_report"] = generate_report
    except Exception as e:
        logger.error(f"report import failed: {e}")

    try:
        from modules.domain_detector import detect_domain, check_domain_mismatch
        mods["detect_domain"] = detect_domain
        mods["check_domain_mismatch"] = check_domain_mismatch
    except Exception as e:
        logger.error(f"domain_detector import failed: {e}")

    try:
        from modules.multi_domain_analyzer import analyze_multi_domain
        mods["analyze_multi_domain"] = analyze_multi_domain
    except Exception as e:
        logger.error(f"multi_domain_analyzer import failed: {e}")

    loaded = list(mods.keys())
    logger.info(f"Loaded {len(loaded)} modules: {loaded}")
    return mods

M = _try_import()

# ── Lifespan (startup/shutdown) ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background session cleanup on startup; cancel on shutdown."""
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    logger.info("[startup] Periodic session cleanup started (every %ds)", CLEANUP_INTERVAL_SECONDS)
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("[shutdown] Cleanup task stopped")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PreTrainAudit",
    version="1.1.0",
    description="Dataset risk intelligence platform",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ── Serve frontend ─────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx), media_type="text/html")
    return HTMLResponse("<h2>PreTrainAudit API is running. Open frontend/index.html in your browser.</h2>")

# ── Session store with TTL ─────────────────────────────────────────────────────
sessions: dict = {}


def _now_utc() -> datetime:
    """Return current UTC time as a timezone-aware datetime (Python 3.12+ safe)."""
    return datetime.now(timezone.utc)


def _cleanup_expired_sessions() -> int:
    """Remove expired sessions and their files. Returns count of cleaned sessions."""
    now = _now_utc()
    expired = [
        sid for sid, sess in sessions.items()
        if sess.get("expires_at") and datetime.fromisoformat(sess["expires_at"]) < now
    ]

    cleaned = 0
    for sid in expired:
        try:
            Path(sessions[sid]["path"]).unlink(missing_ok=True)
            del sessions[sid]
            cleaned += 1
            logger.info(f"[cleanup] Removed expired session {sid}")
        except Exception as e:
            logger.error(f"[cleanup] Failed to remove session {sid}: {e}")

    return cleaned


async def _periodic_cleanup():
    """Background task to periodically clean up expired sessions."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        cleaned = _cleanup_expired_sessions()
        if cleaned > 0:
            logger.info(f"[cleanup] Cleaned {cleaned} expired sessions")


# ── Valid values for enum-like fields ──────────────────────────────────────────
VALID_AI_TYPES      = {"NLP", "Computer Vision", "Recommendation", "Speech", "General"}
VALID_MODEL_SIZES   = {"Small", "Medium", "Large"}
VALID_ARCHITECTURES = {"Transformer", "CNN", "RNN", "MoE"}
VALID_DOMAINS       = {
    "Healthcare", "Finance", "Social Media", "General",
    "Computer Vision", "Music/Audio", "E-Commerce", "NLP/Text",
}
VALID_TASK_TYPES    = {"Classification", "Generation", "Detection", "Recommendation"}


# ── Pydantic models ────────────────────────────────────────────────────────────
class AIConfig(BaseModel):
    ai_type:              str            = "NLP"
    model_size:           str            = "Medium"
    architecture:         str            = "Transformer"
    domain:               str            = "General"
    task_type:            str            = "Classification"
    param_count_millions: Optional[float] = 7000.0

    @field_validator("ai_type")
    @classmethod
    def validate_ai_type(cls, v: str) -> str:
        if v not in VALID_AI_TYPES:
            raise ValueError(f"Invalid ai_type '{v}'. Valid: {sorted(VALID_AI_TYPES)}")
        return v

    @field_validator("model_size")
    @classmethod
    def validate_model_size(cls, v: str) -> str:
        if v not in VALID_MODEL_SIZES:
            raise ValueError(f"Invalid model_size '{v}'. Valid: {sorted(VALID_MODEL_SIZES)}")
        return v

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        if v not in VALID_ARCHITECTURES:
            raise ValueError(f"Invalid architecture '{v}'. Valid: {sorted(VALID_ARCHITECTURES)}")
        return v

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        if v not in VALID_DOMAINS:
            raise ValueError(f"Invalid domain '{v}'. Valid: {sorted(VALID_DOMAINS)}")
        return v

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        if v not in VALID_TASK_TYPES:
            raise ValueError(f"Invalid task_type '{v}'. Valid: {sorted(VALID_TASK_TYPES)}")
        return v


class SimulationConfig(BaseModel):
    session_id: str
    ai_config:  AIConfig


class DomainCheckRequest(BaseModel):
    user_selected_domain: str

    @field_validator("user_selected_domain")
    @classmethod
    def validate_user_selected_domain(cls, v: str) -> str:
        if v not in VALID_DOMAINS:
            raise ValueError(f"Invalid domain '{v}'. Valid: {sorted(VALID_DOMAINS)}")
        return v


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "version":        "1.1.0",
        "sessions":       len(sessions),
        "modules_loaded": list(M.keys()),
        "upload_dir":     str(UPLOAD_DIR),
    }


# ── Security helpers ───────────────────────────────────────────────────────────
def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    name = Path(filename).name
    name = "".join(c for c in name if ord(c) >= 32)
    if len(name) > 255:
        name = name[-255:]
    if not name or name.startswith("."):
        name = "data"
    return name


# ── Upload (streaming to avoid RAM load) ──────────────────────────────────────
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...), ttl_seconds: Optional[int] = None):
    """Upload and ingest a dataset file.

    File is streamed directly to disk (1 MB chunks) to avoid loading large
    files into RAM. Sessions expire after ttl_seconds (default: 1 hour).
    """
    if "ingest_dataset" not in M:
        raise HTTPException(500, "Ingestion module failed to load — check server logs")

    # Validate extension before touching disk
    raw_name = file.filename or "data.csv"
    suffix   = Path(raw_name).suffix.lower()
    allowed  = {".csv", ".parquet", ".arrow", ".xlsx", ".xls", ".txt", ".jsonl", ".json"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {sorted(allowed)}")

    session_id = str(uuid.uuid4())
    save_path  = UPLOAD_DIR / f"{session_id}{suffix}"

    # Stream to disk in 1 MB chunks
    bytes_written = 0
    try:
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    f.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"File too large ({bytes_written // (1024**2)} MB). "
                        f"Max: {MAX_UPLOAD_BYTES // (1024**2)} MB."
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Could not save file to disk: {e}")

    if bytes_written == 0:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty (0 bytes)")

    logger.info(f"[{session_id}] Saved: {file.filename} ({bytes_written // 1024} KB) → {save_path}")

    # Ingest
    try:
        summary = M["ingest_dataset"](str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"[{session_id}] Ingestion error: {e}")
        raise HTTPException(400, f"Could not parse file: {e}")

    # Session metadata — FIX: timezone-aware UTC timestamps
    ttl        = ttl_seconds or SESSION_TTL_SECONDS
    created_at = _now_utc()                              # ← was datetime.utcnow()
    expires_at = created_at + timedelta(seconds=ttl)

    sessions[session_id] = {
        "path":        str(save_path),
        "filename":    file.filename,
        "summary":     summary,
        "created_at":  created_at.isoformat(),
        "expires_at":  expires_at.isoformat(),
        "ttl_seconds": ttl,
    }

    logger.info(
        f"[{session_id}] Ingested: {summary.get('data_type')} — "
        f"{summary.get('row_count', summary.get('line_count', '?'))} rows/lines "
        f"(expires in {ttl}s)"
    )
    return {"session_id": session_id, "summary": summary, "expires_at": expires_at.isoformat()}


# ── Profile ────────────────────────────────────────────────────────────────────
@app.get("/profile/{session_id}")
async def profile(session_id: str, sample_size: int = 100_000):
    sess   = _get_session(session_id)
    _require("profile_dataset")
    result = _run(lambda: M["profile_dataset"](sess["path"], sample_size=sample_size), "Profiling failed")
    sessions[session_id]["profile"] = result
    return result


@app.get("/quality/{session_id}")
async def quality(session_id: str):
    sess   = _get_session(session_id)
    _require("compute_quality_score", "profile_dataset")
    prof   = sess.get("profile") or M["profile_dataset"](sess["path"])
    result = M["compute_quality_score"](prof)
    sessions[session_id]["quality"] = result
    return result


# ── Domain Detection ───────────────────────────────────────────────────────────
@app.get("/domain/{session_id}")
async def detect_dataset_domain(session_id: str):
    sess   = _get_session(session_id)
    _require("detect_domain", "profile_dataset")
    prof   = sess.get("profile") or M["profile_dataset"](sess["path"])
    result = M["detect_domain"](prof, sess["summary"])
    sessions[session_id]["domain_detection"] = result
    sessions[session_id]["profile"]          = prof
    return result


@app.post("/domain/check/{session_id}")
async def check_user_domain(session_id: str, body: DomainCheckRequest):
    sess      = _get_session(session_id)
    _require("detect_domain", "check_domain_mismatch", "profile_dataset")
    prof      = sess.get("profile") or M["profile_dataset"](sess["path"])
    detection = sess.get("domain_detection") or M["detect_domain"](prof, sess["summary"])
    mismatch  = M["check_domain_mismatch"](detection, body.user_selected_domain)
    sessions[session_id]["domain_detection"] = detection
    sessions[session_id]["domain_mismatch"]  = mismatch
    sessions[session_id]["profile"]          = prof
    return {"domain_detection": detection, "mismatch_check": mismatch}


# ── Multi-Domain ───────────────────────────────────────────────────────────────
@app.post("/analyze/multi_domain/{session_id}")
async def multi_domain(session_id: str, sample_size: int = 200_000):
    sess = _get_session(session_id)
    _require("analyze_multi_domain")
    ext = Path(sess["path"]).suffix.lower()
    if ext not in (".csv", ".parquet"):
        raise HTTPException(400, f"Multi-domain analysis supports CSV and Parquet only. Got: {ext}")
    result = _run(
        lambda: M["analyze_multi_domain"](sess["path"], sess["summary"], sample_size),
        "Multi-domain analysis failed",
    )
    sessions[session_id]["multi_domain"] = result
    return result


# ── Risk Analysis ──────────────────────────────────────────────────────────────
@app.post("/analyze/adversarial/{session_id}")
async def adversarial(session_id: str):
    sess   = _get_session(session_id)
    _require("scan_adversarial_vulnerability")
    result = _run(
        lambda: M["scan_adversarial_vulnerability"](sess["path"], sess["summary"]),
        "Adversarial scan failed",
    )
    sessions[session_id]["adversarial"] = result
    return result


@app.post("/analyze/memorization/{session_id}")
async def memorization(session_id: str, param_count_millions: float = 7000.0):
    sess   = _get_session(session_id)
    _require("compute_memorization_risk")
    result = M["compute_memorization_risk"](sess["summary"], param_count_millions)
    sessions[session_id]["memorization"] = result
    return result


@app.post("/analyze/fairness/{session_id}")
async def fairness(session_id: str):
    sess   = _get_session(session_id)
    _require("predict_fairness_drift", "profile_dataset")
    prof   = sess.get("profile") or M["profile_dataset"](sess["path"])
    result = M["predict_fairness_drift"](prof)
    sessions[session_id]["fairness"] = result
    return result


@app.post("/analyze/contamination/{session_id}")
async def contamination(session_id: str):
    sess   = _get_session(session_id)
    _require("detect_benchmark_contamination")
    result = _run(
        lambda: M["detect_benchmark_contamination"](sess["path"], sess["summary"]),
        "Contamination scan failed",
    )
    sessions[session_id]["contamination"] = result
    return result


@app.post("/analyze/cost/{session_id}")
async def cost(session_id: str, ai_config: AIConfig):
    sess   = _get_session(session_id)
    _require("estimate_training_cost")
    result = M["estimate_training_cost"](sess["summary"], ai_config.model_dump())
    sessions[session_id]["cost"] = result
    return result


# ── Compatibility ──────────────────────────────────────────────────────────────
@app.post("/compatibility/{session_id}")
async def compatibility(session_id: str, ai_config: AIConfig):
    sess  = _get_session(session_id)
    _require("evaluate_compatibility", "compute_quality_score", "profile_dataset")
    prof  = sess.get("profile") or M["profile_dataset"](sess["path"])
    qual  = sess.get("quality")  or M["compute_quality_score"](prof)

    detection = None
    mismatch  = None
    if "detect_domain" in M and "check_domain_mismatch" in M:
        detection = sess.get("domain_detection") or M["detect_domain"](prof, sess["summary"])
        mismatch  = M["check_domain_mismatch"](detection, ai_config.domain)
        sessions[session_id]["domain_detection"] = detection
        sessions[session_id]["domain_mismatch"]  = mismatch

    result = M["evaluate_compatibility"](prof, qual, ai_config.model_dump())

    if mismatch and mismatch.get("mismatch"):
        result["domain_detection"] = detection
        result["domain_mismatch"]  = mismatch
        result["issues"]           = [mismatch["message"]] + result.get("issues", [])
        if mismatch.get("severity") == "HIGH":
            result["compatibility_score"] = round(result["compatibility_score"] * 0.80, 4)
            result["issues"].append("Compatibility score reduced 20% due to HIGH-severity domain mismatch.")

    sessions[session_id]["compatibility"] = result
    sessions[session_id]["profile"]       = prof
    sessions[session_id]["quality"]       = qual
    return result


# ── Report ─────────────────────────────────────────────────────────────────────
@app.post("/report/{session_id}")
async def report(session_id: str, ai_config: AIConfig):
    sess   = _get_session(session_id)
    _require("generate_report", "profile_dataset", "compute_quality_score", "evaluate_compatibility")
    prof   = sess.get("profile")       or M["profile_dataset"](sess["path"])
    qual   = sess.get("quality")       or M["compute_quality_score"](prof)
    compat = sess.get("compatibility") or M["evaluate_compatibility"](prof, qual, ai_config.model_dump())

    result = M["generate_report"](
        summary       = sess["summary"],
        profile       = prof,
        quality       = qual,
        adversarial   = sess.get("adversarial", {}),
        memorization  = sess.get("memorization", {}),
        fairness      = sess.get("fairness", {}),
        contamination = sess.get("contamination", {}),
        cost          = sess.get("cost", {}),
        compatibility = compat,
        ai_config     = ai_config.model_dump(),
        filename      = sess["filename"],
    )

    result["domain_intelligence"] = {
        "detection":      sess.get("domain_detection"),
        "mismatch_check": sess.get("domain_mismatch"),
        "multi_domain":   sess.get("multi_domain"),
    }
    return result


# ── Simulate ───────────────────────────────────────────────────────────────────
@app.post("/simulate")
async def simulate(config: SimulationConfig):
    sess  = _get_session(config.session_id)
    _require(
        "evaluate_compatibility", "estimate_training_cost",
        "compute_memorization_risk", "profile_dataset", "compute_quality_score",
    )
    prof  = sess.get("profile") or M["profile_dataset"](sess["path"])
    qual  = sess.get("quality")  or M["compute_quality_score"](prof)
    cfg   = config.ai_config.model_dump()

    mismatch = None
    if "detect_domain" in M and "check_domain_mismatch" in M:
        detection = sess.get("domain_detection") or M["detect_domain"](prof, sess["summary"])
        mismatch  = M["check_domain_mismatch"](detection, config.ai_config.domain)

    compat = M["evaluate_compatibility"](prof, qual, cfg)
    if mismatch and mismatch.get("mismatch") and mismatch.get("severity") == "HIGH":
        compat["compatibility_score"] = round(compat["compatibility_score"] * 0.80, 4)

    return {
        "compatibility":   compat,
        "cost":            M["estimate_training_cost"](sess["summary"], cfg),
        "memorization":    M["compute_memorization_risk"](
                               sess["summary"],
                               config.ai_config.param_count_millions or 7000.0,
                           ),
        "domain_mismatch": mismatch,
    }


# ── Session management ─────────────────────────────────────────────────────────
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    return _get_session(session_id)


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    sess = _get_session(session_id)
    Path(sess["path"]).unlink(missing_ok=True)
    del sessions[session_id]
    return {"deleted": session_id}


# ── Internal helpers ───────────────────────────────────────────────────────────
def _get_session(sid: str) -> dict:
    if sid not in sessions:
        raise HTTPException(404, f"Session '{sid}' not found. Please re-upload your dataset.")
    return sessions[sid]


def _require(*names: str):
    missing = [n for n in names if n not in M]
    if missing:
        raise HTTPException(500, f"Required module(s) not loaded: {missing}. Check server logs.")


def _run(fn, msg: str):
    try:
        return fn()
    except Exception as e:
        logger.error(f"{msg}: {e}")
        raise HTTPException(500, f"{msg}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
