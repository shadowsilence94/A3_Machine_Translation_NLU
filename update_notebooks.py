import json
import os

files = [
    'Attention_Experiments.ipynb',
    'German_English_Transformer.ipynb'
]
base_path = '/Users/htutkoko/Library/CloudStorage/GoogleDrive-htutkoko1994@gmail.com/My Drive/NLP/Project_A3/A3_Burmese_English_Puffer/'

target_line = "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
new_line = "device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))"

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
                    if target_line in line:
                        new_source.append(line.replace(target_line, new_line))
                        modified = True
                    else:
                        new_source.append(line)
                cell['source'] = new_source
        
        if modified:
            with open(path, 'w') as file:
                json.dump(data, file, indent=1) # indent=1 to keep file size reasonable but readable
            print(f"Updated {f}")
        else:
            print(f"No changes needed for {f}")
    else:
        print(f"File {f} not found")
