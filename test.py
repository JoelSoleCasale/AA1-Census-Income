import json

def print_notebook_cells_and_outputs(file_name, ni, nf):
    ni -= 1
    nf -= 1
    with open(file_name, 'r') as f:
        nb = json.load(f)

        for i, cell in enumerate(nb['cells']):
            if i < ni:
                continue
            if i > nf:
                break

            # if all the cell code is commented out, skip it
            if cell['cell_type'] == 'code':
                # check if all the code is comented
                all_code_commented = True
                for line in cell['source']:
                    if not line.startswith('#'):
                        all_code_commented = False
                        break
                if all_code_commented:
                    continue

            if cell['cell_type'] == 'code':
                print(f"\nCell {i+1}:")
                code_text = ''.join(cell['source'])
                print(code_text)
            elif cell['cell_type'] == 'markdown':
                print(f"\nMarkdown {i+1}:")
                markdown_text = ''.join(cell['source'])
                print(markdown_text)

            if 'outputs' in cell:
                if cell['outputs']:
                    print("Output:")

                for output in cell['outputs']:
                    if 'text' in output:
                        output_text = ''.join(output['text'])
                        print(output_text)

                    if output.get('output_type') == 'execute_result':
                        data = output['data']
                        for mime_type in ['text/plain', 'image/png', 'image/jpeg']:
                            if mime_type in data:
                                output_text = ''.join(data[mime_type])
                                print(output_text)

if __name__ == '__main__':
    print("""Describe and analyse the outputs in a formal style for a section of a machine learning paper the following code and its output.
do not describe the code itself, only the output and the method used to obtain it, as well as the results obtained.""")

    file_name = 'aa1_project.ipynb'
    ni = 18
    nf = 37
    print_notebook_cells_and_outputs(file_name, ni, nf)
