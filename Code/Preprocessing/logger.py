import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Fonction pour les messages d'erreur
def error(message):
    logging.error(message)

# Fonction pour les messages d'information
def info(message):
    logging.info(message)

# Fonction pour les messages d'avertissement
def warning(message):
    logging.warning(message)

# Fonction pour les messages de débogage
def debug(message):
    logging.debug(message)
