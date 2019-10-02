import os

city='sherbrooke'
suffix='leftImg8bit'

#changes current working directory
path=os.chdir(input('Dans quel dossier sont les photos a convertir? '))

underscores = True
if underscores:
    num_index=int((input('When split with \'_\', what index in number?' )))

print ("Current working directory is: %s"%(os.getcwd()))

nb_files=len(os.listdir('.'))
print('Number of files in source directory: ', str(nb_files))

for i in os.listdir('.'):
    try:
        file, ext = os.path.splitext(i)  # separate file and extension (ex.: .png)
        if underscores:
            elements = file.split('_')
            #for l and r images
            beg = '_'.join(elements[:num_index])
            #orig_filename=elements[num_index] #get number with extension of full image. ex.: 5.jpg
            #num=orig_filename.split('.')[0] #get only number from that string
            num = elements[num_index] #if not 5.jpg in middle of filename (for l and r images for example)
            end = '_'.join(elements[(num_index+1):])
            #new_name = f'{city}_{int(num):04d}_'+end+ext
            if num_index != 0:
                new_name = f'{beg}_{int(num):04d}_{end}{ext}'
            else:
                new_name = f'{int(num):04d}_{end}{ext}'
        else:
            num = file
            new_name = f'{int(num):04d}{ext}'
        os.rename(i,new_name)
        #file, ext = os.path.splitext(i)  # separate file and extension (ex.: .png)
        # .save(f"{city}_{int(file):04d}_{left}_{suffix}.png")
    except:
        pass
