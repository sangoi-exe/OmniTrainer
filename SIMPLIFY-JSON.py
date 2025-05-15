import json
from pathlib import Path

# 1) Carrega o JSON original
orig_path = Path("todos_modulos_d_num_den_compressed.json")
data = json.loads(orig_path.read_text())

# 2) Comprime: para cada bloco, monta s, e, n e d
compressed = {}
for bloco, entradas in data.items():
    s_list, e_list, n_list, d_list = [], [], [], []
    for ent in entradas:
        # extrai intervalo (steps ou step)
        rng = ent.get("steps") or str(ent.get("step"))
        if "-" in rng:
            a, b = rng.split("-")
            start, end = int(a), int(b)
        else:
            start = end = int(rng)
        s_list.append(start)
        e_list.append(end)
        n_list.append(ent["d_num"])
        d_list.append(ent["d_den"])
    compressed[bloco] = {"s": s_list, "e": e_list, "n": n_list, "d": d_list}

# 3) Prepara divisão em ~40 chaves por arquivo
total_keys = len(compressed)  # ex.: 247
desired_per_file = 1
num_files = max(1, total_keys // desired_per_file)
remainder = total_keys % desired_per_file

# monta lista de quantas chaves cada parte terá
counts = [desired_per_file] * num_files
for i in range(remainder):
    counts[i % num_files] += 1

# 4) Gera cada parte e salva minificado
keys = list(compressed.keys())
start = 0
out_dir = Path("parts")
out_dir.mkdir(exist_ok=True)

for idx, count in enumerate(counts, start=1):
    part_keys = keys[start:start + count]
    part_data = {k: compressed[k] for k in part_keys}
    fn = out_dir / f"todos_modulos_part_{idx:02d}.json"
    fn.write_text(json.dumps(part_data, separators=(",", ":")))
    print(f"→ {fn.name}: {count} chaves")
    start += count
