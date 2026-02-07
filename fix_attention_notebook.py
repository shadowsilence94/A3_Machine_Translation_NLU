import json
import os

files = [
    'Attention_Experiments.ipynb'
]
base_path = '/Users/htutkoko/Library/CloudStorage/GoogleDrive-htutkoko1994@gmail.com/My Drive/NLP/Project_A3/A3_Burmese_English_Puffer/'

replacements = [
    ("input = trg[:, 0]", "input = trg[0, :]"),
    ("input = trg[:, t]", "input = trg[t, :]")
]

for f in files:
    path = os.path.join(base_path, f)
    if os.path.exists(path):
        with open(path, 'r') as file:
            data = json.load(file)
        
        modified = False
        for cell in data['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    original_line = line
                    for old, new in replacements:
                        if old in line:
                            line = line.replace(old, new)
                    
                    if line != original_line:
                        modified = True
                        new_source.append(line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source
        
        if modified:
            with open(path, 'w') as file:
                json.dump(data, file, indent=1)
            print(f"Fixed bug in {f}")
        else:
            print(f"No changes needed for {f}")
    else:
        print(f"File {f} not found")
