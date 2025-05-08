from rich.console import Console as RichConsole  # Renomear para clareza
from typing import Optional

# Cria um console global padrão para logFun, se nenhum específico for passado
_default_logfun_console = RichConsole()
_current_logfun_console = _default_logfun_console

def set_logfun_console(console: RichConsole) -> None:
    """
    Redefine o console que o logFun vai usar por padrão.
    Deve ser chamado antes de instanciar o Live, ex:
        set_logfun_console(meu_trainer.console)
    """
    global _current_logfun_console
    _current_logfun_console = console


def logFun(mensagem: str, lvl: str = "INFO", _console: Optional[RichConsole] = None) -> None:
    """
    Imprime uma mensagem de log colorida no console usando match-case.

    Args:
        mensagem (str): A mensagem a ser exibida.
        lvl (str): O nível do log ('INFO', 'VERBOSE', 'WARNING', 'ERROR', 'DEBUG', 'SUCCESS', 'LOOP', 'CONVCTRL', 'TRAINGPS', 'LORA').
                   Determina a cor da mensagem.
        _console (Optional[RichConsole]): O console Rich a ser usado. Se None, usa um console padrão.
    """
    console_to_use = _console or _current_logfun_console
    level_upper = lvl.upper()

    match level_upper:
        case "INFO":
            console_to_use.print(f"[dark_olive_green1][INFO][/dark_olive_green1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "LOOP":  # Usado pelo Trainer
            console_to_use.print(f"[cyan][TRAINER][/cyan] [sky_blue1]{mensagem}[/sky_blue1]")
        case "CONVCTRL":  # Específico para ConvergeControl
            console_to_use.print(f"[light_salmon1][CONVCTRL][/light_salmon1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "TRAINGPS":
            console_to_use.print(f"[light_steel_blue3][TRAINGPS][/light_steel_blue3] [sky_blue1]{mensagem}[/sky_blue1]")
        case "LORA":
            console_to_use.print(f"[slate_blue1][LORA][/slate_blue1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "VERBOSE":
            console_to_use.print(f"[orange3][VERBOSE][/orange3] [sky_blue1]{mensagem}[/sky_blue1]")
        case "WARNING":
            console_to_use.print(f"[gold1][WARNING][/gold1] [sky_blue1]{mensagem}[/sky_blue1]")
        case "ERROR":
            console_to_use.print(f"[red][ERROR][/red] [pink1]{mensagem}[/pink1]",
                                 soft_wrap=True)  # Adicionado soft_wrap
            # Adicionar traceback aqui se desejado, condicionalmente
            # import traceback
            # console_to_use.print_exception(show_locals=True) # Ou False para menos verbosidade
        case "DEBUG":
            console_to_use.print(f"[grey35][DEBUG][/grey35] [sky_blue1]{mensagem}[/sky_blue1]")
        case "SUCCESS":
            console_to_use.print(f"[spring_green3][SUCCESS][/spring_green3] [sky_blue1]{mensagem}[/sky_blue1]")
        case _:
            console_to_use.print(f"[white][{level_upper}][/white] [sky_blue1]{mensagem}[/sky_blue1]")
