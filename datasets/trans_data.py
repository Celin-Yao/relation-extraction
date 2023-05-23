import json 

def get_all_data():
    filename = "./CMeIE_train.json"
    rel = []
    tmp = []
    fin_rel = []
    D = []
    fin_D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 0):
            line = json.loads(line)
            D.append({
                "text":line["text"],
                "spo_list":[(spo["subject"], spo["object"]["@value"], spo["predicate"])
                            for spo in line["spo_list"]]
            })
            rel.append({"spo_list":[spo["predicate"] for spo in line["spo_list"]]}) 
    with open('all_data.txt','w', encoding='utf-8') as fp:
        for i in range(len(D)):
            for j in range(len(D[i]['spo_list'])):
                fp.write(str(D[i]['spo_list'][j][0]) + '\t')
                fp.write(str(D[i]['spo_list'][j][1]) + '\t')
                fp.write(str(D[i]['spo_list'][j][2]) + '\t')
                fp.write(D[i]['text'] + '\n')
    print(fin_D)
    for i in range(len(rel)):
        tmp.append(rel[i]['spo_list'])
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            fin_rel.append(tmp[i][j])
    fin_rel = set(fin_rel)
    with open('relation.txt','w', encoding='utf-8') as fp:
        [fp.write(str(item)+'\n') for item in fin_rel]
        fp.close()

get_all_data()

