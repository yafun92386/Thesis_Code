import os, json

rel_path = "dataset/"
sel_data = "train"

input_file = open(rel_path+'DRCD_'+sel_data+'.json', 'r' , encoding = 'utf-8', errors='ignore')
json_data = json.load(input_file)

data_array = json_data['data']

# for data array
for each_title in data_array:
    
    # data['title']
    folder_name = each_title['title']
    folder_path = rel_path +sel_data+"_set/" + folder_name
    try:
        os.mkdir(folder_path)
    except FileExistsError:
        print("Folder already exists.")

    # data['paragraphs']
    paragraphs = each_title['paragraphs']

    # for paragraphs array
    for each_paragraph in paragraphs:

        # paragraphs['id']
        file_name = each_paragraph['id']
        file_path = folder_path +"/"+ file_name + ".txt"
        
        file_ptr = open(file_path, 'w', encoding = 'utf-8', errors='ignore')

        # paragraphs['context']
        context = each_paragraph['context']

        if context[-1]!="。":
            file_ptr.write(context+"。")
        else:
            file_ptr.write(context)
        file_ptr.write("\n")

        # each_paragraph['qas']
        qas = each_paragraph['qas']

        # for qas array
        for each_qas in qas:

            # qas['question']
            question = each_qas['question']
            if question[-1]!="?" & question[-1]!="？":
                file_ptr.write(question+"?")
            else:
                file_ptr.write(question+"#")

            # qas['answers']
            answers = each_qas['answers']

            # for answer array
            for each_ans in answers:

                # answer['text']
                text = each_ans['text']
                file_ptr.write(text+"#")
                # answer['answer_start']
                pos = each_ans['answer_start']
                file_ptr.write(str(pos))

                break

            file_ptr.write("\n")

        file_ptr.close()

input_file.close()





