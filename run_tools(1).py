#!/usr/bin/env python3
"""
run_tools.py

1. Build seqs.csv from therasab.csv + JSON files in jsons/
   (drugs that have at least one ground truth property).
2. Build antibody PDB structures using ABodyBuilder2 into
   /scratch/users/peckmann/ab_structures/ (existing structures are reused).
3. For each tool overlay in /scratch/users/peckmann/apptainer_tools/overlays/:
   a. Stage seqs.csv and PDB files into the overlay home/ directory.
   b. Remove any existing output CSV so the job starts fresh.
   c. Write a per-tool sbatch file.
   d. Submit with sbatch.
4. Print a summary of submitted jobs and exit.

Run via sbatch (see run_tools.sbatch) — ABodyBuilder2 needs GPU resources.
"""

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

import pandas as pd

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=None,
                    help='Randomly sample N antibodies for testing. Omit to run all.')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
JSONS_DIR     = os.path.join(SCRIPT_DIR, 'jsons')
SEQS_CSV      = os.path.join(SCRIPT_DIR, 'seqs.csv')
THERASAB_CSV  = os.path.join(SCRIPT_DIR, 'therasab.csv')
AB_STRUCTURES = '/scratch/users/peckmann/ab_structures'
OVERLAYS_DIR  = '/scratch/users/peckmann/apptainer_tools/overlays'
BASE_SANDBOX  = '/scratch/users/peckmann/apptainer_tools/base/sandbox'
SBATCH_DIR    = os.path.join(SCRIPT_DIR, 'tool_sbatch')
LOGS_DIR      = os.path.join(SCRIPT_DIR, 'tool_logs')

INPUT_CSV_NAME = 'seqs.csv'

# ---------------------------------------------------------------------------
# Step 1: Load JSON files
# ---------------------------------------------------------------------------

print('=== Loading ground truth from jsons/ ===')
json_files = sorted(glob.glob(os.path.join(JSONS_DIR, '*.json')))
if not json_files:
    sys.exit(f'No JSON files found in {JSONS_DIR}')

records = []
json_extras = []
drug_affinity_target: dict[str, str] = {}

for jf in json_files:
    with open(jf) as fh:
        d = json.load(fh)
    if 'drug_name' not in d:
        continue
    records.append({
        'drug_name':               d['drug_name'],
        'immunogenicity_rate_pct': d.get('immunogenicity_rate_pct'),
        'sae_rate_pct':            d.get('sae_rate_pct'),
        'neutralizing_ab_pct':     d.get('neutralizing_ab_pct'),
        'half_life_hours':         d.get('half_life_hours'),
        'n_patients_tested':       d.get('n_patients_tested'),
    })
    aff = d.get('binding_affinity_nM')
    aff_vals = [v for v in aff.values() if isinstance(v, (int, float)) and v > 0] if aff else []
    json_extras.append({
        'drug_name':          d['drug_name'],
        'min_affinity_nM':    min(aff_vals) if aff_vals else None,
        'chembl_heavy_chain': d.get('chembl_heavy_chain') or '',
        'chembl_light_chain': d.get('chembl_light_chain') or '',
    })
    if aff:
        for tgt_str in aff.keys():
            if tgt_str.strip():
                drug_affinity_target[d['drug_name']] = tgt_str
                break

all_drugs      = pd.DataFrame(records)
json_extras_df = pd.DataFrame(json_extras)
print(f'Total drugs in jsons: {len(all_drugs)}')

# Keep drugs with at least one ground truth property
GT_COLS = ['immunogenicity_rate_pct', 'sae_rate_pct', 'neutralizing_ab_pct', 'half_life_hours']
ground_truth = all_drugs.dropna(subset=GT_COLS, how='all').copy()
print(f'Drugs with at least one ground truth property: {len(ground_truth)}')

# ---------------------------------------------------------------------------
# UniProt helper
# ---------------------------------------------------------------------------

def _fetch_uniprot_sequence(gene_aliases: list[str]) -> str:
    base = 'https://rest.uniprot.org/uniprotkb/search'
    for alias in gene_aliases:
        alias = alias.strip()
        if not alias:
            continue
        query = f'gene_exact:{alias} AND organism_id:9606 AND reviewed:true'
        url = f'{base}?query={urllib.request.quote(query)}&fields=sequence&format=json&size=1'
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read())
            results = data.get('results', [])
            if results:
                seq = results[0].get('sequence', {}).get('value', '')
                if seq:
                    return seq
        except Exception:
            pass
        time.sleep(0.15)
    return ''

# ---------------------------------------------------------------------------
# Step 2: Build seqs.csv
# ---------------------------------------------------------------------------

print('\n=== Building seqs.csv ===')

if os.path.isfile(SEQS_CSV):
    print(f'Found existing seqs.csv — loading cached version.')
    seqs = pd.read_csv(SEQS_CSV)
    print(f'Loaded {len(seqs)} rows.')
else:
    therasab = pd.read_csv(THERASAB_CSV)
    merged = therasab[['Therapeutic', 'HeavySequence', 'LightSequence']].copy()
    merged = merged.rename(columns={'Therapeutic': 'drug_name'})
    merged = merged[
        merged['HeavySequence'].notna() | merged['LightSequence'].notna()
    ].copy()
    print(f'Drugs with at least one sequence: {len(merged)}')

    merged = merged.merge(
        json_extras_df[['drug_name', 'chembl_heavy_chain', 'chembl_light_chain']],
        on='drug_name', how='left',
    )

    print('Fetching antigen sequences from UniProt (human reviewed)...')
    antigen_map: dict[str, str] = {}
    drugs_needing_antigen = [
        name for name in merged['drug_name'].tolist()
        if name in drug_affinity_target
    ]
    for drug_name in drugs_needing_antigen:
        tgt_str = drug_affinity_target[drug_name]
        aliases = [a.strip() for a in re.split(r'[/;]', tgt_str) if a.strip()]
        seq = _fetch_uniprot_sequence(aliases)
        if seq:
            antigen_map[drug_name] = seq
        else:
            print(f'  [WARN] No UniProt sequence for {drug_name} (target: {tgt_str})')
    print(f'Retrieved antigen sequences for {len(antigen_map)} / {len(drugs_needing_antigen)} drugs.')

    merged['antigen'] = merged['drug_name'].map(antigen_map).fillna('')

    seqs = (
        merged
        .rename(columns={
            'drug_name':          'name',
            'HeavySequence':      'cdr_heavy_chain',
            'LightSequence':      'cdr_light_chain',
            'chembl_heavy_chain': 'full_heavy_chain',
            'chembl_light_chain': 'full_light_chain',
        })
        [['name', 'cdr_heavy_chain', 'cdr_light_chain',
          'full_heavy_chain', 'full_light_chain', 'antigen']]
        .fillna({'full_heavy_chain': '', 'full_light_chain': '', 'antigen': ''})
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    seqs.to_csv(SEQS_CSV, index=False)
    print(f'Written: {SEQS_CSV}')

print(f'seqs.csv: {len(seqs)} drugs')

if args.n is not None:
    if args.n >= len(seqs):
        print(f'[WARN] --n {args.n} >= total rows ({len(seqs)}); using all.')
    else:
        seqs = seqs.sample(n=args.n, random_state=42).reset_index(drop=True)
        print(f'Subsampled to {len(seqs)} antibodies (--n {args.n})')

# ---------------------------------------------------------------------------
# Step 3: Build antibody PDB structures using ABodyBuilder2
# ---------------------------------------------------------------------------

print(f'\n=== Building antibody PDB structures ===')
os.makedirs(AB_STRUCTURES, exist_ok=True)

from ImmuneBuilder import ABodyBuilder2  # noqa: E402 — loaded after paths are set
predictor = ABodyBuilder2()

n_built  = 0
n_skipped = 0
n_failed = 0
for _, row in seqs.iterrows():
    name = str(row['name'])
    pdb_path = os.path.join(AB_STRUCTURES, f'{name}.pdb')

    if os.path.isfile(pdb_path):
        n_skipped += 1
        continue

    heavy = str(row.get('full_heavy_chain') or '').strip()
    light = str(row.get('full_light_chain') or '').strip()

    if not heavy:
        print(f'  [SKIP] {name}: no full heavy chain sequence')
        n_failed += 1
        continue

    sequences: dict[str, str] = {'H': heavy}
    if light:
        sequences['L'] = light

    try:
        antibody = predictor.predict(sequences)
        antibody.save(pdb_path)
        n_built += 1
        if n_built % 20 == 0:
            print(f'  Built {n_built} structures...')
    except Exception as exc:
        print(f'  [ERROR] {name}: {exc}')
        n_failed += 1

print(f'Built {n_built} structures; {n_skipped} already existed (skipped); {n_failed} failed/skipped.')

# ---------------------------------------------------------------------------
# Step 4: Find overlay tools
# ---------------------------------------------------------------------------

print(f'\n=== Finding tools in {OVERLAYS_DIR} ===')

overlay_dirs = sorted(
    d.rstrip('/') for d in glob.glob(os.path.join(OVERLAYS_DIR, '*/'))
    if os.path.isdir(os.path.join(d.rstrip('/'), 'home'))
    and os.path.isfile(os.path.join(d.rstrip('/'), 'env.conf'))
)

if not overlay_dirs:
    sys.exit(f'No tool overlays found under {OVERLAYS_DIR}')

tool_names = [os.path.basename(d) for d in overlay_dirs]
print(f'Found {len(overlay_dirs)} tools: {tool_names}')

# ---------------------------------------------------------------------------
# Step 5: Set up and submit one Slurm job per tool
# ---------------------------------------------------------------------------

os.makedirs(SBATCH_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

pdb_files = [f for f in os.listdir(AB_STRUCTURES) if f.endswith('.pdb')]

submitted: list[tuple[str, str]] = []
errors:    list[str]             = []

for overlay_dir in overlay_dirs:
    tool_name    = os.path.basename(overlay_dir)
    overlay_home = os.path.join(overlay_dir, 'home')
    overlay_tmp  = os.path.join(overlay_dir, 'tmp')
    overlay_env  = os.path.join(overlay_dir, 'env.conf')
    output_csv   = os.path.join(overlay_home, f'{tool_name}_out.csv')

    print(f'\n--- Setting up {tool_name} ---')

    os.makedirs(overlay_home, exist_ok=True)

    # Stage seqs.csv (the active/possibly-subsampled version)
    staged_input = os.path.join(overlay_home, INPUT_CSV_NAME)
    seqs.to_csv(staged_input, index=False)
    print(f'  Staged seqs.csv ({len(seqs)} rows)')

    # Stage PDB structures
    n_staged = 0
    for fname in pdb_files:
        src = os.path.join(AB_STRUCTURES, fname)
        dst = os.path.join(overlay_home, fname)
        shutil.copy2(src, dst)
        n_staged += 1
    print(f'  Staged {n_staged} PDB files')

    # Remove existing output CSV so the job starts fresh
    if os.path.isfile(output_csv):
        os.remove(output_csv)
        print(f'  Removed existing output CSV')

    # Build apptainer run command (matching install_tools.py format)
    input_csv_ctr  = f'/root/{INPUT_CSV_NAME}'
    output_csv_ctr = f'/root/{tool_name}_out.csv'
    run_cmd = (
        f'apptainer exec --containall --nv '
        f'--overlay {overlay_dir} '
        f'--bind {overlay_tmp}:/tmp '
        f'--home {overlay_home}:/root '
        f'--env-file {overlay_env} '
        f'{BASE_SANDBOX} '
        f'python3 /root/predict.py {input_csv_ctr} {output_csv_ctr}'
    )

    # Write per-tool sbatch file
    sbatch_path = os.path.join(SBATCH_DIR, f'run_{tool_name}.sbatch')
    log_out = os.path.join(LOGS_DIR, f'{tool_name}_%j.out')
    log_err = os.path.join(LOGS_DIR, f'{tool_name}_%j.err')

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=run_tools
#SBATCH --time=2-00:00:00
#SBATCH -p jamesz,owners
#SBATCH --gres=gpu:1
#SBATCH -C "[GPU_MEM:80GB|GPU_MEM:141GB]"
#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH --output={log_out}
#SBATCH --error={log_err}

module load cuda/12.4.0

{run_cmd}
"""
    with open(sbatch_path, 'w') as fh:
        fh.write(sbatch_content)

    # Submit
    result = subprocess.run(['sbatch', sbatch_path], capture_output=True, text=True)
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f'  Submitted job {job_id}')
        submitted.append((tool_name, job_id))
    else:
        msg = result.stderr.strip() or result.stdout.strip()
        print(f'  [ERROR] sbatch failed: {msg}')
        errors.append(tool_name)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f'\n{"="*60}')
print(f'Done. Submitted {len(submitted)} job(s):')
for tool_name, job_id in submitted:
    print(f'  {tool_name:30s}  job {job_id}')
if errors:
    print(f'\nFailed to submit {len(errors)} tool(s): {errors}')
print('\nCheck job status:  squeue --me')
print(f'Tool logs:         {LOGS_DIR}/')
print(f'Output CSVs will appear in: {OVERLAYS_DIR}/<tool>/home/<tool>_out.csv')
