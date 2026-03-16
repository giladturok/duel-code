import csv
import os
import numpy as np

dir_name = 'sample_logs'

# report averages
for fname in os.listdir(dir_name):
  with open(f'{dir_name}/{fname}', 'r') as f:
    fieldnames = ['gen_ppl', 'nfes', 'entropy', 'gen_lengths', 'mauve', 'samples', 'seed']
    reader = csv.DictReader(f, fieldnames=fieldnames)
    rows = list(reader)
    if len(rows) > 1000:
      rows = rows[-1000:]
    filtered_rows = [row for row in rows if float(eval(row['entropy'])) >= 1.0]
    gen_ppl = [float(eval(row['gen_ppl'])) for row in filtered_rows if row['gen_ppl'] != '[nan]']
    try:
      nfes = [float(eval(row['nfes'])) for row in filtered_rows]
    except:
      nfes = [0 for row in filtered_rows]
    entropy = [float(eval(row['entropy'])) for row in filtered_rows]
    gen_lengths = [float(eval(row['gen_lengths'])) for row in filtered_rows]
    mauve = [float(eval(row['mauve'])) for row in filtered_rows]
    
    print(f'{fname}:')
    print(f'Average over {len(gen_ppl)} samples\n')
    print(f'Generative Perplexity: {sum(gen_ppl) / len(gen_ppl)}')
    print(f'NFEs: {sum(nfes) / len(nfes)}')
    print(f'Entropy: {sum(entropy) / len(entropy)}')
    print(f'Median entropy: {np.median(entropy)}')
    print(f'Median sample length: {np.median(gen_lengths)}')
    print(f'Max sample length: {max(gen_lengths)}')
    print(f'MAUVE: {sum(mauve) / len(mauve)}')
    print('\n-----------------------\n')
