import logging
from rich.console import Console
from rich.logging import RichHandler

# crie o Console que você vai usar no seu Live
console = Console()

# configure o handler para usar esse console
rich_handler = RichHandler(console=console, show_time=False, markup=True)
logging.basicConfig(level=logging.INFO, handlers=[rich_handler])

# opcional: um logger “global” da sua aplicação
logger = logging.getLogger("sangoi")
