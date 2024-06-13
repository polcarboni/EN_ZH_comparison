import os

def import_split_stories_dict(directory_name):

    result_dict = {}

    for filename in os.listdir(directory_name):
        file_path = os.path.join(directory_name, filename)

        with open(file_path, 'r') as file:
            content = file.read()

        string_list = content.split(" ")
        string_list.pop()
        key = os.path.splitext(filename)[0] #get key from filename
        key = key.split("_")[0]

        result_dict[key] = string_list

    return result_dict