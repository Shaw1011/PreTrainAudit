from .ingestion            import ingest_dataset, get_dataset_summary
from .profiler             import profile_dataset
from .quality              import compute_quality_score
from .adversarial          import scan_adversarial_vulnerability
from .memorization         import compute_memorization_risk
from .fairness             import predict_fairness_drift
from .contamination        import detect_benchmark_contamination
from .cost_estimator       import estimate_training_cost
from .compatibility        import evaluate_compatibility
from .report               import generate_report
from .domain_detector      import detect_domain, check_domain_mismatch
from .multi_domain_analyzer import analyze_multi_domain
