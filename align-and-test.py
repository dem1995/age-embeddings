
import gensim
from gensim.models import Word2Vec
import pickle
from procrustes_align import smart_procrustes_align_gensim
from sklearn.metrics.pairwise import cosine_similarity


from gensim.test.utils import datapath
from gensim import utils


firstmodelindex = 1
secondmodelindex = 3

with open(f"model-pickles/bucket{firstmodelindex}.pickle", 'rb') as filein:
	firstmodel = pickle.load(filein)

with open(f"model-pickles/bucket{secondmodelindex}.pickle", 'rb') as filein:
	secondmodel = pickle.load(filein)

# with open("model-pickles/bucket1.pickle", 'rb') as filein:
# 	model1 = pickle.load(filein)

# with open("model-pickles/bucket3.pickle", 'rb') as filein:
# 	model3 = pickle.load(filein)

firstmodel_aligned = firstmodel
secondmodel_aligned = smart_procrustes_align_gensim(firstmodel, secondmodel)


# model0_aligned = model0
# model3_aligned = smart_procrustes_align_gensim(model0, model3)
# model1 = model0_aligned
# model2 = model3_aligned

# print(model.wv.most_similar(['friend']))

# m1wv = model0_aligned.wv
# m2wv = model3_aligned.wv
m1wv = firstmodel_aligned.wv
m2wv = secondmodel_aligned.wv


vocab1 = set(m1wv.index2word)
vocab2 = set(m2wv.index2word)

print(len(vocab1))
print(len(vocab2))

vocabinboth = vocab1 & vocab2



cosinesims = dict()
for word in vocabinboth:
	if (m1wv.vocab[word].count>50 and m2wv.vocab[word].count>50
		and len(word)>3):
		vec1 = (firstmodel_aligned[word]).reshape(1, -1)
		vec2 = (secondmodel_aligned[word]).reshape(1, -1)
		cosinesims[word] = cosine_similarity(vec1, vec2)

sortedsimilarities = sorted(cosinesims.items(), key=lambda x: x[1])

for cosinesim in sortedsimilarities[0:20]:
	print("-----------------------------")
	print(cosinesim)
	print()
	print("Most similar for younger")
	print(m1wv.most_similar([cosinesim[0]]))
	print("Most similar for older")
	print(m2wv.most_similar([cosinesim[0]]))
	print("-----------------------------")

cosinesim = cosinesims['thesis']
print(cosinesim)
print(f"Most similar for younger group, group {firstmodelindex}")
print(m1wv.most_similar(['exam']))
print("Most similar for older group, group ")
print(m2wv.most_similar(['exam']))
print("-----------------------------")