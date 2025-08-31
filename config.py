#!/usr/bin/env python3
import os, warnings, logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 120

USE_LLM_EXPLAINER = os.getenv("USE_LLM_EXPLAINER", "1") == "1"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = ModelConfig().model
LLM_MAX_TOKENS = ModelConfig().max_tokens

def _merge_secrets_and_env():
    """Load .env (if present), merge Streamlit secrets, sanitize, and reflect into os.environ."""
    # 1) Load .env (requires python-dotenv)
    try:
        from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)
            logging.getLogger(__name__).info(f".env loaded from: {dotenv_path}")
    except Exception as e:
        logging.getLogger(__name__).warning(f"python-dotenv not available or failed: {e}")

    # 2) Merge Streamlit secrets → environment (without overwriting anything already set)
    try:
        candidates = [
            Path.home() / ".streamlit" / "secrets.toml",
            Path.cwd() / ".streamlit" / "secrets.toml",
        ]
        if any(p.exists() for p in candidates):
            import streamlit as st
            secrets = st.secrets  # safe to access now
            for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "NEOS_EMAIL"):
                if key in secrets and not os.getenv(key):
                    os.environ[key] = str(secrets[key]).strip()
    except Exception:
        pass

    # 3) Trim whitespace on the keys we care about
    for key in ("GROQ_API_KEY", "NEOS_EMAIL"):
        val = os.getenv(key)
        if isinstance(val, str):
            os.environ[key] = val.strip()

# Call it before Config() materializes
_merge_secrets_and_env()

@dataclass
class SystemPaths:
    root_dir: Path = Path(__file__).parent
    output_dir: Path = root_dir / "output_data"

    # scripts live at repo root
    scripts_dir: Path = root_dir

    # canonical files – everything lands in output_data/
    synthetic_data_file: Path = output_dir / "synthetic_ops_data_monthly.csv"
    optimization_results_dir: Path = output_dir            # keep results flat in output_data/
    optimization_results_file: Path = output_dir / "opti_results_latest.csv"
    conversations_dir: Path = output_dir / "conversations"
    logs_dir: Path = root_dir / "logs"

    def __post_init__(self):
        for d in [self.output_dir, self.optimization_results_dir, self.conversations_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

@property
def is_ready_for_autogen(self) -> bool:
    """Check if system is ready for AutoGen"""
    return (
        self._groq_api_key is not None and 
        self._groq_api_key != "your_groq_api_key_here" and
        len(self._groq_api_key) > 10
    )

class Config:
    def __init__(self):
        self.paths = SystemPaths()
        self.model_config = ModelConfig()
        self._groq_api_key: Optional[str] = None
        self._neos_email: Optional[str] = None
        self._initialization_errors: List[str] = []
        self.setup_environment()

    def setup_environment(self):
        try:
            # read AFTER _merge_secrets_and_env() so .env/secrets are visible
            self._groq_api_key = os.getenv("GROQ_API_KEY")
            if not self._groq_api_key:
                self._initialization_errors.append(
                    "GROQ_API_KEY not set. Get one at https://console.groq.com/keys"
                )
                logger.warning("GROQ_API_KEY not set")
            else:
                # reflect back into env so any code that uses os.getenv(...) sees it
                os.environ["GROQ_API_KEY"] = self._groq_api_key

            self._neos_email = os.getenv("NEOS_EMAIL")
            if self._neos_email:
                os.environ["NEOS_EMAIL"] = self._neos_email

            warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
        except Exception as e:
            self._initialization_errors.append(f"Environment setup failed: {e}")
            logger.error(f"Environment setup failed: {e}")

    @property
    def groq_api_key(self) -> Optional[str]:
        return self._groq_api_key

    @property
    def neos_email(self) -> str:
        return self._neos_email or "demo@example.com"

    @property
    def has_errors(self) -> bool:
        return bool(self._initialization_errors)

    @property
    def errors(self) -> List[str]:
        return self._initialization_errors.copy()

    def get_groq_config(self) -> Dict[str, Any]:
        if not self._groq_api_key:
            raise ValueError("GROQ API key not available")
        return {
            "model": self.model_config.model,
            "api_key": self._groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "api_type": "openai",
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "timeout": self.model_config.timeout,
        }

    def get_autogen_llm_config(self) -> List[Dict[str, Any]]:
        try:
            return [self.get_groq_config()]
        except ValueError as e:
            logger.error(f"Cannot create AutoGen config: {e}")
            return []

    @property
    def default_scenario_template(self) -> Dict[str, Any]:
        return {
            "scope": {"skus": ["*"], "locations": ["*"]},
            "changes": {
                "service_level_target": {"default": 0.95, "overrides": {}},
                "lead_time_days": {"multiplier": 1.0, "by_supplier": {}},
                "moq_units": {"overrides": {}},
                "review_period_days": 1
            },
            "run_options": {"scenario_name": "baseline", "seed": 42},
        }

try:
    config = Config()
    logger.info("Configuration initialized successfully")
except Exception as e:
    logger.error(f"Configuration initialization failed: {e}")
    config = None

def validate_dependencies() -> Dict[str, Any]:
    """Non-fatal dependency report (used by Streamlit)."""
    required_packages = {
        "streamlit": "streamlit", "pandas": "pandas", "numpy": "numpy",
        "pyomo": "pyomo", "pyautogen": "autogen", "groq": "groq",
        "plotly": "plotly", "watchdog": "watchdog"
    }
    results = {"missing": [], "available": [], "versions": {}, "success": True}
    for pkg, import_name in required_packages.items():
        try:
            m = __import__(import_name.replace("-", "_"))
            results["available"].append(pkg)
            results["versions"][pkg] = getattr(m, "__version__", "unknown")
        except ImportError as e:
            results["missing"].append(pkg)
            results["success"] = False
            logger.warning(f"Missing package {pkg}: {e}")
    if results["missing"]:
        results["error_message"] = f"Missing required packages: {', '.join(results['missing'])}"
    return results

def get_system_prompt_base() -> str:
    return (
        "You are an expert supply chain optimization assistant with deep knowledge of:\n"
        "- Inventory management and reorder point optimization\n"
        "- Safety stock calculations and service level targets\n"
        "- Lead time variability and demand forecasting\n"
        "- Cost optimization and trade-off analysis\n\n"
        "Always provide clear, actionable insights backed by quantitative analysis.\n"
        "Be concise but thorough."
    )

def safe_config_access() -> 'Config':
    global config
    if config is None:
        logger.warning("Using fallback configuration")
        return Config()
    return config

def check_system_health() -> Dict[str, Any]:
    """Comprehensive system health check (works with output_data/)"""
    import pandas as pd
    health = {
        "timestamp": str(pd.Timestamp.now()),
        "config_loaded": config is not None,
        "groq_api_configured": False,
        "data_directory_exists": False,
        "dependencies": {},
        "errors": [],
        "warnings": []
    }
    # Data file readable?
    try:
        if config is not None:
            health["groq_api_configured"] = bool(config.groq_api_key)
            health["data_directory_exists"] = config.paths.output_dir.exists()
            health["errors"].extend(config.errors)
        else:
            health["data_file_readable"] = False
    except Exception as e:
        health["data_file_readable"] = False
        health["warnings"].append(f"Data file not readable: {e}")
    health["status"] = "healthy" if not health["errors"] else "degraded"
    return health
