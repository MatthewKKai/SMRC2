import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
import json

# define PCA
estimator = PCA(n_components=3)

# abs1 = r'his review explores similarities between lymphocytes and cancer cells, and proposes a new model for the genesis of human cancer. we suggest that the development of cancer requires infection(s) during which antigenic determinants from pathogens mimicking self-antigens are co-presented to the immune system, leading to breaking t cell tolerance. some level of autoimmunity is normal and necessary for effective pathogen eradication. however, autoreactive t cells must be eliminated by apoptosis when the immune response is terminated. apoptosis can be deficient in the event of a weakened immune system, the causes of which are multifactorial. some autoreactive t cells suffer genomic damage in this process, but manage to survive. the resulting cancer stem cell still retains some functions of an inflammatory t cell, so it seeks out sites of inflammation inside the body. due to its defective constitutive production of inflammatory cytokines and other growth factors, a stroma is built at the site of inflammation similar to the temporary stroma built during wound healing. the cancer cells grow inside this stroma, forming a tumor that provides their vascular supply and protects them from cellular immune response.</p><p>as cancer stem cells have plasticity comparable to normal stem cells, interactions with surrounding normal tissues cause them to give rise to all the various types of cancers, resembling differentiated tissue types. metastases form at an advanced stage of the disease, with the proliferation of sites of inflammation inside the body following a similar mechanism. immunosuppressive cancer therapies inadvertently re-invigorate pathogenic microorganisms and parasitic infections common to cancer, leading to a vicious circle of infection, autoimmunity and malignancy that ultimately dooms cancer patients. based on this new understanding, we recommend a systemic approach to the development of cancer therapies that supports rather than antagonizes the immune system.'
# abs2 = r'given the fundamental principle that cancer must arise from a cell that has the potential to divide, two major nonexclusive hypotheses of the cellular origin of cancer are that malignancy arises a) from stem cells due to maturation arrest or b) from dedifferentiation of mature cells that retain the ability to proliferate. the role of stem cells in carcinogenesis is clearly demonstrated in teratocarcinomas. the malignant stem cells of teratocarcinomas are derived from normal multipotent stem cells and have the potential to differentiate into normal benign mature tissue. a widely studied model supporting dedifferentiation has been the putative origin of hepatocarcinomas from "premalignant" foci and nodules induced in the rat liver by chemicals. however, the dedifferentiation concept for hepatocarcinogenesis is challenged by more recent interpretations indicating that hepatocellular carcinoma arises from maturation arrest caused by aberrant differentiation of determined stem cells. either hypothesis is supported by the cellular changes that occur in the rodent liver after different hepatocarcinogenic regimens. the formation of foci and nodules from altered hepatocytes supports dedifferentiation; the proliferation of small oval cells with the potential to differentiate into either biliary ducts or hepatocytes supports arrested maturation of determined stem cells. it is now postulated that foci and nodular change reflect adaptive changes to the toxic effects of carcinogens and not "preneoplastic" stages to cancer. the stem cell model predicts that genotoxic chemicals induce mutations in the determined stem cell which may be expressed in its progeny. proliferation of initiated cells is induced by promoting events which also allow additional mutations to occur.'
# abs3 = r'medical histories of themselves and their first-degree relatives were obtained from parents of 82 leukaemic children (54 acute lymphoblastic (all), 28 acute myeloblastic (aml)) and from control couples matched for age. the possibility of a primary familial immunological abnormality as an aetiological factor in childhood leukaemia was suggested by binding some infections significantly more frequently reported in parents than in controls, but more strongly supported by the finding of a significantly (p less than 0.02) increased prevalence of disorders associated with autoimmunity (but not of other conditions such as peptic ulceration, infective hepatitis, tuberculosis or malignancy) amongst members of all families compared to those of controls. analogy with down\'s syndrome and the strain of nzb mice, in which diminished t-cell function is associated with autoimmune disease and lymphoid neoplasia, is discussed. varicella and herpes zoster occurred respectively in 2 all mothers during their pregnancies involving the patients and in none of the other 388 pregnancies here reported. this supports previous evidence that antenatal varicella infections may be of aetiological importance in some cases of all.'
#
# s1 = r"I love apple"
# s2 = [abs1, abs2, abs3]

# label -> color
labels2color = {'cardiovascular diseases': 'red',
'chronic kidney disease': 'purple',
'chronic respiratory diseases': 'grey',
'ciabetes mellitus': 'white',
'cigestive diseases': 'blue',
'hiv/aids': 'black',
'hepatitis a/b/c/e': 'green',
'mental disorders': 'yellow',
'musculoskeletal disorders': 'pink',
'neoplasms (cancer)': 'orange',
'neurological disorders': 'brown'}


# load data
with open(r"data.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

text = []
color = []
for i in range(100):
    temp = data[i]['abs']
    text.append(temp)
    color_transit = data[i]['label'][0]
    color.append(labels2color[color_transit])

print(len(text))


# biobert dmis-lab/biobert-base-cased-v1.2
bert_model = BertModel.from_pretrained(r"dmis-lab/biobert-base-cased-v1.2")
tokenizer = BertTokenizer.from_pretrained(r"dmis-lab/biobert-base-cased-v1.2")

# tokens_s1 = tokenizer(s1,truncation=True, padding=True, return_tensors="pt")
tokens_s2 = tokenizer(text, max_length=512, truncation=True, padding=True, return_tensors="pt")
# output1 = bert_model(**tokens_s1)
output2 = bert_model(**tokens_s2)

# print("s1 is:{}".format(output1.pooler_output))
# print("s2 is:{}".format(output2))
# print(output1.pooler_output.shape)
# print(output2.pooler_output.shape)

# test distribution estimator
# output1_tensor = output1.pooler_output
output2_tensor = output2.pooler_output

print(output2_tensor.detach().numpy().shape)

# using pca
pca_output = estimator.fit_transform(output2_tensor.detach().numpy())
print(pca_output.shape)
# print(pca_output)

# kde1 = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(output1_tensor.detach().numpy())
# kde2 = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(output1)
# print(output1_tensor.detach().numpy().reshape(-1, 2))
fig = plt.figure()
ax = Axes3D(fig)
xx = pca_output[..., 0]
print(len(xx))
yy = pca_output[..., 1]
zz = pca_output[..., 2]

ax.scatter(xx, yy, zz, c=color)
plt.show()