import pubmed_parser as pp
import os
import asyncio
from tqdm import tqdm
import json

root = r'../../raw_data/'

# print(os.listdir(root_path))

'''
Obtain data content: 
{'paper A': {'pmid':xxxx, 'abs': xxx, 'entity_set': [], 'label': xxx}, 
 'paper B': {'pmid':xxxx, 'abs': xxx, 'entity_set': [], 'label': xxx}, 
 'paper C': {'pmid':xxxx, 'abs': xxx, 'entity_set': [], 'label': xxx},
 'paper D': {'pmid':xxxx, 'abs': xxx, 'entity_set': [], 'label': xxx},
 'paper E': {'pmid':xxxx, 'abs': xxx, 'entity_set': [], 'label': xxx},
 'citation': [(pmid, pmid), (pmid, pmid)]}
'''

'''
Pre-defined 11 class diseases: ['Cardiovascular diseases',
'Chronic kidney disease', 
'Chronic respiratory diseases',
'Diabetes mellitus',
'Digestive diseases',
'HIV/AIDS',
'Hepatitis A/B/C/E',
'Mental disorders',
'Musculoskeletal disorders',
'Neoplasms (cancer)',
'Neurological disorders']
'''

labels = ['cardiovascular diseases',
'chronic kidney disease',
'chronic respiratory diseases',
'ciabetes mellitus',
'cigestive diseases',
'hiv/aids',
'hepatitis a/b/c/e',
'mental disorders',
'musculoskeletal disorders',
'neoplasms (cancer)',
'neurological disorders']

async def get_data(root_path):
    all_data = []
    print(root_path)
    file_path_list = os.listdir(root_path)
    print(file_path_list)
    for file_path in file_path_list:
        print("-------------Processing File: {}------------\n".format(file_path))
        try:
            raw_data = pp.parse_medline_xml(os.path.join(root_path, file_path))
        except Exception as e:
            print(e)
            continue
        temp = await get_content(raw_data)
        print("\nfinished {} processing".format(file_path))
        all_data.extend(temp)
    return all_data

async def get_content(raw_data):
    data = []
    with tqdm(total=len(raw_data)) as pbar:
        pbar.set_description("Data Processing:")
        for i in range(len(raw_data)):
            pmid = get_pmid(raw_data[i])
            abs = get_abs(raw_data[i])
            label = get_MeSH_terms(raw_data[i])
            references = get_citation(raw_data[i])
            # print(label)
            if label:
                if isEmpty(abs) or isEmpty(references):
                    continue
                else:
                    temp = {'pmid': pmid, 'abs': abs, 'references': references, 'label': label}
                    # print(temp)
                    data.append(temp)
                pbar.update(1)
            else:
                pbar.update(1)
                continue
    return data

def get_pmid(data_i):
    return data_i["pmid"]

def get_abs(data_i):
    return data_i["abstract"]

def get_MeSH_terms(data_i):
    mesh_terms = data_i["mesh_terms"].split(";")
    terms = []
    if len(mesh_terms)==0:
        return None
    else:
        for i in range(len(mesh_terms)):
            try:
                terms.append(mesh_terms[i].split(":")[1].lower())
            except Exception as e:
                print(e)
                # print(mesh_terms)
                continue
        # print(terms)
        intersection = list(set(terms) & set(labels))
        # print(intersection)
        if intersection:
            return intersection
        else:
            return None

def get_citation(data_i):
    references = data_i['references']
    return references

# empty or not
def isEmpty(str):
    if len(str)==0:
        return True
    else:
        return False

if __name__=="__main__":
    data = asyncio.run(get_data(root))
    print(len(data))

    # print(data)
    with open(r"data.json", "w", encoding="utf-8") as f:
        json.dump(data, f)