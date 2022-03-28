labels = []
labels.append(open(f"Labels_swe/Adina_labels.txt", "r",encoding="utf-8").readlines())
labels.append(open(f"Labels_swe/Christian_labels.txt", "r",encoding="utf-8").readlines())
labels.append(open(f"Labels_swe/Liza_labels.txt", "r",encoding="utf-8").readlines())
labels.append(open(f"Labels_swe/Sofia_labels.txt", "r",encoding="utf-8").readlines())

labels_scrab = open(f"Labels_swe/Scrab_labels.txt", "r",encoding="utf-8").readlines()


createOrNot = open(f"createdOrNot.txt", "r").readlines()
createOrNot = [int(number.strip()) for number in createOrNot]

name_roots = ["swe-ad-ad","swe-ch-ch","swe-li-li","swe-so-so"]
name_roots_scrab = ["aug-style_1-","aug-style_2-","aug-style_5-","aug-style_7-"]

f = open('words_swe.txt','w')
p=0
for k in range(len(name_roots)):
    for i in range(len(labels[0])):
        filenames = []
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_di")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_er")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_ds")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_dss")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_es")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_ess")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_s")
        p+=1
        if createOrNot[p]:
            filenames.append(f"{name_roots[k]}{f'{i+1:03}'}_ss")
        p+=1
        

        for j in range(len(filenames)):
            f.write(f"{filenames[j]} ok 123 123 123 123 123 ABC {labels[k][i]}")

offset = 0
for k in range(len(name_roots_scrab)):
    for i in range(len(labels_scrab)+offset):
        filename = []
        try:
            filename.append(f"{name_roots_scrab[k]}{f'{i:05}'}")
        except FileNotFoundError:
            print("No such file")
            offset+=1

        f.write(f"{filename[0]} ok 123 123 123 123 123 ABC {labels_scrab[i-offset]}")


name_roots_samp = ["samples-ad","samples-ch","samples-er"]
sample_lengths = [33, 21, 34]
fs = open('samples.txt','w')
for k in range(len(name_roots_samp)):
    for i in range(sample_lengths[k]):
        fs.write(f"{name_roots_samp[k]}-{i+1} ok 123 123 123 123 123 ABC \n")


