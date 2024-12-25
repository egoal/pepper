import json


def load_json(file: str):
    with open(file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    return data


def make_json(jsonfile: str, data):
    with open(jsonfile, 'w') as fout:
        json.dump(data, fout, indent=4)
    return jsonfile


def write_to_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as fin:
        fin.write(content)
