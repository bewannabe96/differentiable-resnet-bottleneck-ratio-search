from os import walk
import csv
import re

files = []
for (dirpath, dirnames, filenames) in walk('./csv'):
    files = filenames
    break

train_files = []
val_files = []
for f in files:
    if re.search('val', f):
        val_files.append(f)
    else:
        train_files.append(f)

ratios = {}
for f in val_files:
    ratios[int(f[9])] = f[12:19]

train_csv_readers = []
val_csv_readers = []
for i in range(len(ratios)):
    trn_f = 'csv/csv_trial%d_r%s.train.csv' % (i, ratios[i])
    val_f = 'csv/csv_trial%d_r%s.val.csv' % (i, ratios[i])

    train_csv_readers.append(csv.reader(open(trn_f, 'r', newline='\n')))
    val_csv_readers.append(csv.reader(open(val_f, 'r', newline='\n')))

with open('output.csv', 'w', newline='\n') as f_output:
    csv_output = csv.writer(f_output)

    new_row = []
    for j in range(len(ratios)):
        new_row += [ratios[j], ratios[j], ratios[j]]
        new_row += [ratios[j], ratios[j], ratios[j]]
    csv_output.writerow(new_row)
    
    new_row = []
    for j in range(len(ratios)):
        new_row += ['train', 'train', 'train']
        new_row += ['val', 'val', 'val']
    csv_output.writerow(new_row)

    new_row = []
    for j in range(len(ratios)):
        new_row += ['time', 'loss', 'acc']
        new_row += ['time', 'loss', 'acc']
    csv_output.writerow(new_row)

    for i in range(100):
        new_row = []
        for j in range(len(ratios)):
            new_row += next(train_csv_readers[j])[1:4]
            new_row += next(val_csv_readers[j])[1:4]
        csv_output.writerow(new_row)