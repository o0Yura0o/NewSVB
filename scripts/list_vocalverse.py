from huggingface_hub import list_repo_files

files = list(list_repo_files('karl-wang/VocalVerse-dataset', repo_type='dataset'))
print(f'Total files: {len(files)}')

# Show non-wav files
non_wav = [f for f in files if not f.endswith('.wav')]
print(f'\nNon-WAV files ({len(non_wav)}):')
for f in non_wav:
    print(f'  {f}')

# Count unique top-level dirs
import os
dirs = set()
for f in files:
    parts = f.split('/')
    if len(parts) > 1:
        dirs.add(parts[0])
print(f'\nTop-level directories ({len(dirs)}):')
for d in sorted(dirs):
    count = sum(1 for f in files if f.startswith(d + '/'))
    print(f'  {d}/  ({count} files)')
