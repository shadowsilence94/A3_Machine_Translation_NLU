import json
import os

files = [
    'Attention_Experiments.ipynb'
]
base_path = '/Users/htutkoko/Library/CloudStorage/GoogleDrive-htutkoko1994@gmail.com/My Drive/NLP/Project_A3/A3_Burmese_English_Puffer/'

# Fix trg_len bug
search_str = "trg_len = trg.shape[1]"
replace_str = "trg_len = trg.shape[0]"

# Colab Setup Cell
colab_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Google Colab Setup\n",
        "try:\n",
        "    import google.colab\n",
        "    IN_COLAB = True\n",
        "    print(\"Running in Google Colab\")\n",
        "    !pip install datasets sentencepiece\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    # Optional: Change to your project directory if needed\n",
        "    # import os\n",
        "    # os.chdir('/content/drive/MyDrive/NLP/Project_A3/A3_Burmese_English_Puffer')\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "    print(\"Running Locally\")"
    ]
}

targets = ['Attention_Experiments.ipynb', 'German_English_Transformer.ipynb']

for f in targets:
    path = os.path.join(base_path, f)
    if os.path.exists(path):
        with open(path, 'r') as file:
            data = json.load(file)
        
        # Add Colab cell if not present
        has_colab = False
        if len(data['cells']) > 0 and 'Google Colab Setup' in "".join(data['cells'][0]['source']):
             has_colab = True
        
        if not has_colab:
             data['cells'].insert(0, colab_cell)
             print(f"Added Colab setup to {f}")

        # Fix bug in Attention nb
        if f == 'Attention_Experiments.ipynb':
            modified = False
            for cell in data['cells']:
                if cell['cell_type'] == 'code':
                    new_source = []
                    for line in cell['source']:
                        if search_str in line:
                            new_source.append(line.replace(search_str, replace_str))
                            modified = True
                        else:
                            new_source.append(line)
                    cell['source'] = new_source
            if modified:
                print(f"Fixed trg_len bug in {f}")

        with open(path, 'w') as file:
            json.dump(data, file, indent=1)
    else:
        print(f"File {f} not found")
