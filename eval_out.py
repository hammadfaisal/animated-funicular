from utils_gpt import data_loader
import json
import sys, pdb, os
from tqdm import tqdm
import re
SOURCE_PATH = sys.argv[1]
OUT_FILE = sys.argv[2]
REL_FILE = "benchmark/nyt10m/nyt10m_rel2id.json"
all_rels = {}

with open(REL_FILE, "r") as f:
    all_rels = json.load(f)

all_rels = {}

with open(REL_FILE, "r") as f:
    all_rels = json.load(f)

#all_rels = {os.path.basename(x): y for x, y in all_rels.items()}
    # all_rels = list(data.keys())

delim_str = "[{'role': 'system', 'content'"

all_preds = []
with open(OUT_FILE, "r") as f:
    all_lines = f.readlines()  
    all_blocks = [[]]
    for l in tqdm(all_lines, desc="Processing lines"):
        if l.strip().startswith(delim_str) and len(all_blocks[-1]) > 0:
            all_blocks.append([])
        all_blocks[-1].append(l)
    # print(all_blocks[0])
    # print(all_blocks[0])
    for block in tqdm(all_blocks, desc="Processing blocks"):
        print(block[3:])
        pred_lines = [x.strip() for x in block[3:] if x.strip() != ""]
        p1 = []
        for l in pred_lines:
            p1.extend(re.split(r'[ ,]+', l))
        # print(p1)
        pred = []
        for k, v in all_rels.items():
            for l in p1:
                if k == l:
                    pred.append(k)
                    break
        all_preds.append(pred)
        # print(pred)
# print(all_preds[-1]
# print(len(all_preds))
# all_preds = all_preds[:-1]
# print(all_preds[0])
# exit()

source_data = []
with open(SOURCE_PATH, "r") as f:
    for l in f:
        source_data.append(json.loads(l))

print(len(source_data))
print(len(all_preds))
#pdb.set_trace()
id_cnt = 0

pred_utils_format = []
source_utils_format = []
for idx, s in enumerate(source_data):
    s["h"] = {"id": str(id_cnt)}
    id_cnt += 1
    s["t"] = {"id": str(id_cnt)}
    id_cnt += 1
    #pdb.set_trace()
    #s["relations"] = [os.path.basename(x) for x in s["relations"]]
    #pdb.set_trace()
    if "token" in s:
        s["text"] = " ".join(s["token"])
    #print(s["text"])
    #print(s["relations"])
    #print(idx)
    #print(all_preds[idx])
    for r in s["relations"]:
        if (r == "NA"):
            continue
        source_utils_format.append({"text":s["text"], "h": {"id": s["h"]["id"]}, "t": {"id": s["t"]["id"]}, "relation": r})
    #print(all_preds[idx])
    for p in all_preds[idx]:
        pred_utils_format.append({"entpair":(s["h"]["id"], s["t"]["id"]), "relation": p, "score": 1.0})

#pdb.set_trace()
dataset = data_loader.PassageRELoader(source_utils_format, all_rels)
results = dataset.eval(pred_utils_format)
print("Micro F1 ", results["micro_f1"])
print("Macro F1 ", results["macro_f1"])



