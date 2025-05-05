#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------
#  RESUMO JSON DE CONVERGÊNCIA
#  - Mantém as 10 primeiras amostras de cada módulo
#  - A partir daí, guarda 1 amostra a cada 100 steps
#
#  Uso:
#     python shrink_conv_history.py  input.json  output.json
#
#  Requisitos: Python ≥3.8
# ----------------------------------------------------------

import argparse
import json
import pathlib
from typing import Any, Dict, List

# --------------- START: shrink_conv_history.py ---------------

def resumir_history(lista_steps: List[List[Any]]) -> List[List[Any]]:
    """Retorna nova lista com 10 primeiras amostras + 1 a cada 100 steps."""
    if len(lista_steps) <= 10:
        return lista_steps[:]               # nada a resumir
    head        = lista_steps[:10]
    tail_sample = [row for row in lista_steps[10:] if row[0] % 100 == 0]
    return head + tail_sample


def processa_arquivo(entrada: pathlib.Path, saida: pathlib.Path) -> None:
    print(f"🔄 Lendo {entrada} …")
    data: Dict[str, Any] = json.loads(entrada.read_text())

    conv_hist: Dict[str, List[List[Any]]] = data.get("convergence_history", {})
    print(f"📊 Módulos encontrados: {len(conv_hist)}")

    reduzido = {}
    for modulo, rows in conv_hist.items():
        reduzido[modulo] = resumir_history(rows)

    data["convergence_history"] = reduzido

    print(f"✅ Salvando resumo em {saida}")
    saida.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Cria resumo do JSON de convergência")
    ap.add_argument("input",  help="Arquivo JSON original (≈150 MB)")
    ap.add_argument("output", help="Arquivo JSON resumido")
    args = ap.parse_args()

    processa_arquivo(pathlib.Path(args.input), pathlib.Path(args.output))


if __name__ == "__main__":
    main()

# --------------- END: shrink_conv_history.py ---------------
