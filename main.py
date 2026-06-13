from config.settings import get_settings
from observability.logger import setup_logger

settings = get_settings()
settings.ensure_dirs()

logger = setup_logger(__name__, settings.log_level)
logger.info("LexRAG starting up", extra={"version": "1.0.0"})

from api.app import create_app

app = create_app()
