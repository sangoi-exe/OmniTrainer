import colorama
from colorama import Fore, Style, Back

# Inicializa o colorama (necessário no Windows)
# autoreset=True faz com que cada print() volte à cor padrão automaticamente
colorama.init(autoreset=True)


def logFun(mensagem, lvl="INFO"):
    """
    Imprime uma mensagem de log colorida no console usando match-case.

    Args:
        mensagem (str): A mensagem a ser exibida.
        lvl (str): O nível do log ('INFO', 'VERBOSE', 'WARNING', 'ERROR', 'DEBUG', 'SUCCESS').
                    Determina a cor da mensagem.
    """
    level_upper = lvl.upper()  # Garante que o nível seja maiúsculo para o match

    match level_upper:
        case "INFO":
            print(f"{Fore.CYAN}[INFO] {mensagem}")
        case "VERBOSE":
            print(f"{Fore.LIGHTBLUE_EX}[VERBOSE] {mensagem}")
        case "WARNING":
            print(f"{Fore.YELLOW}[WARNING] {mensagem}")
        case "ERROR":
            print(f"{Fore.RED}[ERROR] {mensagem}")
        case "DEBUG":
            print(f"{Fore.LIGHTBLACK_EX}[DEBUG] {mensagem}")
        case "SUCCESS":
            print(f"{Fore.GREEN}[SUCCESS] {mensagem}")
        case _:  # Caso padrão (default) para níveis não reconhecidos
            # Imprime com a cor padrão (resetada pelo autoreset=True)
            print(f"[{level_upper}] {mensagem}")
