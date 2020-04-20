
import gensim
from gensim.models import Word2Vec
import pickle
from procrustes_align import smart_procrustes_align_gensim
from sklearn.metrics.pairwise import cosine_similarity


from gensim.test.utils import datapath
from gensim import utils


def align_embeddings(firstmodelindex, secondmodelindex, use_sg=1):
    """Align the embeddings of the provided indices and store the alignments to a file"""
    with open(f"model-pickles/bucket{'sg' if use_sg==1 else 'cobw'}{firstmodelindex}.pickle", 'rb') as filein:
        firstmodel = pickle.load(filein)

    with open(f"model-pickles/bucket{'sg' if use_sg==1 else 'cobw'}{secondmodelindex}.pickle", 'rb') as filein:
        secondmodel = pickle.load(filein)

    firstmodel_aligned = firstmodel
    secondmodel_aligned = smart_procrustes_align_gensim(firstmodel, secondmodel)

    m1wv = firstmodel_aligned.wv
    m2wv = secondmodel_aligned.wv

    if firstmodelindex<secondmodelindex:
        ststring = f"{firstmodelindex}{secondmodelindex}"
    else:
        ststring = f"{secondmodelindex}{firstmodelindex}"

    with open(f"aligned-embedding-pickles{'sg' if use_sg==1 else 'cbow'}/embeddings-aligned{ststring}.pickle", 'wb') as embeddingsfile:
        pickle.dump([m1wv, m2wv], embeddingsfile)