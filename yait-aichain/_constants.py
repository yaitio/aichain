import urllib3

# client default timeout is 10 minutes
DEFAULT_TIMEOUT = urllib3.Timeout(connect=10.0, read=600.0)
DEFAULT_MAX_RETRIES = urllib3.Retry(3, redirect=2)
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 8.0

# models constants
DEFAULT_CHATMODEL_TEMPERATURE = 0.75
DEFAULT_MAX_TOKENS = 4096