import logging
from datetime import datetime
import os

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

if not os.path.exists('logs'):
    os.makedirs('logs')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s\n%(message)s\n', filename=f'logs/{now}.log')
