
import os, re
import numpy as np
import pandas as pd

rel_path = "dataset/"
sel_data = "train"

input_path = rel_path+sel_data+"_set/"
output_path = rel_path

short_file = open(output_path+"sS200_"+sel_data+".csv", 'a', encoding = 'utf-8', errors='ignore')
long_file = open(output_path+"lS200_"+sel_data+".csv", 'a', encoding = 'utf-8', errors='ignore')


sentence_len = 0
question_len = 0

count = 0

# input triple data (content) (question answer ans_pos)
for root, dirs, files in os.walk(input_path):
    for f in files:
        # record content & triple_list & answer position
        content = ""
        triple = []
        pos_ans = []

        fullpath = os.path.join(root, f)
        openfile = open(fullpath, 'r' , encoding = 'utf-8', errors='ignore')

        for line in openfile.readlines():
            # first line for content
            if content == "":
                content = line.strip()
                continue
            # down below for triple
            new_line = line.replace(",","")
            part = new_line.split("#")
            if (part[0][-1:]!="?") & (part[0][-1:]!="？"):
                part[0] = part[1]+"?"
            triple.append(part)
            pos_ans.append(int(part[2]))

        # find answer content
        # record comma position
        pattern = [u'。' , u'；']
        regex = re.compile(r'\b(' + '|'.join(pattern) + r')\b')
        pos_comma = [-1]
        pos_find = [m.start() for m in regex.finditer(content)] 
        pos_comma.extend(pos_find)
        pos_comma.extend([len(content)])

        # for answer content
        ans_cont = []

        for a_idx, pos_a in enumerate(pos_ans): 
            for c_idx, pos_c in enumerate(pos_comma):
                if c_idx == len(pos_comma):
                    break
                if pos_a > pos_c:
                    ans_cont.append("")
                    ans_cont[a_idx] = content[pos_comma[c_idx]+1:pos_comma[c_idx+1]+1].replace(",","")
                else:
                    continue

        # rewrite triple data (answer_content question answer)
        for idx in range(len(triple)):
            if len(ans_cont[idx])>200:
                wrtline = ans_cont[idx].replace(",", "")+","+triple[idx][0].replace(",", "")+","+triple[idx][1].replace(",", "")+","+triple[idx][2].replace(",", "")
                long_file.write(wrtline)
            else:
                count += 1
                wrtline = ans_cont[idx].replace(",", "")+","+triple[idx][0].replace(",", "")+","+triple[idx][1].replace(",", "")+","+triple[idx][2].replace(",", "")
                short_file.write(wrtline)
                sentence_len += len(ans_cont[idx])
                question_len += len(triple[idx][0])

print(count)

print(sentence_len)
print(sentence_len/count)

print(question_len)
print(question_len/count)


long_file.close()
short_file.close()

