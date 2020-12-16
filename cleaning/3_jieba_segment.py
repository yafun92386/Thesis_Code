
import re, jieba

rel_path = "dataset/"
data = "train"
datatype = "sS200"

input_file = open(rel_path+datatype+"_"+data+".csv", 'r' , encoding = 'utf-8', errors='ignore')
# clean_file = open(rel_path+"3_segment/"+datatype+"_"+data+"_clean.txt", 'w', encoding = 'utf-8', errors='ignore')
seg_file = open(rel_path+datatype+"_"+data+"_seg.txt", 'w', encoding = 'utf-8', errors='ignore')

jieba_path = r"C:/Users/jieba_dict/"
jieba.set_dictionary(jieba_path+'dict.txt.big')  

def remove_symbol(context):

	# ^\u4E00-\u9FA5 中字 ^\u002e . ^\u0030-\u0039 數字 ^\u002C ,分隔符號
	filtrate = re.compile(u'[^\u4E00-\u9FA5^\u002e^\u0030-\u0039^\u002C]') 
	context = filtrate.sub(r'', context)
	context = context.replace('.','點')

	rep = {'0':'零','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九'}
	rep = dict((re.escape(k), v) for k, v in rep.items())
	pattern = re.compile("|".join(rep.keys()))
	context = pattern.sub(lambda m: rep[re.escape(m.group(0))], context)

	return context


for line in input_file.readlines():

	line = remove_symbol(line)
	# clean_file.write(line+"\n")

	seg_list = jieba.cut(line)

	seg_file.write(" ".join(seg_list))
	seg_file.write("\n")

input_file.close()
# clean_file.close()
seg_file.close()
