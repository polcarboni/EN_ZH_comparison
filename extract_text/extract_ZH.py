from dsutils import make_word_ds
from stimulus_utils import load_grids_for_stories, load_generic_trfiles
import numpy as np

# Set subject to analyze
subject_num = "01"

story_names = ["alternateithicatom", "avatar", "howtodraw", "legacy", "life", "myfirstdaywiththeyankees", "naked", "odetostepfather", "souls", "undertheinfluence", "wheretheressmoke"]
stories_textGrid = []
stories_trfiles = []

for i in range(len(story_names)):
    stories_textGrid.append(story_names[i] + "Audio_en_TYE_0.TextGrid")

for i in range(len(story_names)):
    stories_trfiles.append(story_names[i] + "Audio_en_TYE_0")

grid_dir = "stimulus/grids_EN"
tr_dir = "stimulus/trfiles_EN"

'''
# Set local data directories
# fmri_data_dir = "Deniz2019_subjects01-02/"
# grid_dir = "Deniz2019_grids"
# tr_dir = "Deniz2019_trfiles"

fmri_data_dir = "data/"
'''

print("LOADING: ")
grids = load_grids_for_stories(story_names, grid_dir=grid_dir)
trfiles = load_generic_trfiles(stories_trfiles, root=tr_dir)

#print("SCRIPT: ", grids.keys())
#print("SCRIPT: ", trfiles.keys())

trfiles_2 = {key.replace('Audio_en_TYE_0', ''): value for key, value in trfiles.items()}

wordseqs = make_word_ds(grids, trfiles_2)
print(wordseqs['souls'].data)

print(type(wordseqs))



import os

def save_strings_as_files(data_dict, directory_name):
    # Create a new directory if it doesn't exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    
    # Iterate through the dictionary and save each value as a file
    for key, value in data_dict.items():
        words_list = value.data
        file_name = f"{key}_EN.txt"  # Construct the file name
        file_path = os.path.join(directory_name, file_name)  # Create the full file path
        
        # Write the value to the file
        with open(file_path, 'w') as file:
            for word in words_list:
                file.write(word + " ")

save_strings_as_files(wordseqs, "extracted_text_EN")