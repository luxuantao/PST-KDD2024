import json

files = ['output/test_submission_deberta.json', 'output/test_submission_scibert.json']

final = {}
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for k, v in data.items():
            if k not in final:
                final[k] = [0] * len(v)
            for i in range(len(v)):
                final[k][i] += v[i]

for k, v in final.items():
    final[k] = [each / len(files) for each in v]

with open('ensemble_result.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)
