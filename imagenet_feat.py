import pandas
import numpy
import os

list_of_files = 'hl_all/list_hl.txt'
dir_input = 'hl_all/'
dir1 = 'label_1/'
dir2 = 'label_2/'

#Get list of input files
with open(list_of_files) as f:
    content = f.readlines()

array_list = []

for i in content:
    array_list.append(i.strip('\n'))

#Create output folder
if not os.path.exists('output_imagenet'):
    os.makedirs('output_imagenet')

#Load the imagenet concepts file
df1 = pandas.read_excel('imagenet_concepts.xlsx', sheetname='1stedition', index_col=0, parse_cols=5)
df2 = pandas.read_excel('imagenet_concepts.xlsx', sheetname='2ndedition', index_col=0, parse_cols=5, skip_footer=1)

#Create the output files
for i in array_list:
    filename_in = dir_input+i
    array1 = pandas.read_csv(filename_in, header=None, delimiter=',').values

    if not os.path.exists('output_imagenet/'+i.split('.')[0]):
        os.makedirs('output_imagenet/'+i.split('.')[0])

    dir_output = 'output_imagenet/'+i.split('.')[0]

    for j in df1.columns:

        if not os.path.exists(dir_output+'/'+dir1):
            os.makedirs(dir_output+'/'+dir1)

        filename = dir_output+'/'+dir1+j+'_'+i.split('.')[0]+'_lab1.csv'
        cols= numpy.where(df1[j] == 1)[0]
        df_cat = pandas.DataFrame(array1[:,cols])
        df_cat.to_csv(filename, index=False, header=None)
        #print(j, cols.shape)
        print(filename + ' created')

    print('---')

    for j in df2.columns:

        if not os.path.exists(dir_output+'/'+dir2):
            os.makedirs(dir_output+'/'+dir2)

        filename = dir_output+'/'+dir2+j+'_'+i.split('.')[0]+'_lab2.csv'
        cols= numpy.where(df2[j] == 1)[0]
        df_cat = pandas.DataFrame(array1[:,cols])
        df_cat.to_csv(filename, index=False, header=None)
        #print(j, cols.shape)
        print(filename + ' created')

    print('--------------------------------------------------------------- '+ i+ ' finished')