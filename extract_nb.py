import json
import sys

def extract_notebook_code(ipynb_path, output_py_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            code_cells.append(source)
    
    with open(output_py_path, 'w', encoding='utf-8') as f:
        f.write("\n\n# " + "="*70 + "\n# EXTRACTED CODE CELL\n# " + "="*70 + "\n\n".join(code_cells))

if __name__ == "__main__":
    extract_notebook_code(sys.argv[1], sys.argv[2])
