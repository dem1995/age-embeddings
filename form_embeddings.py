
import gensim
from gensim.models import Word2Vec
import pickle
from procrustes_align import smart_procrustes_align_gensim


from gensim.test.utils import datapath
from gensim import utils

def form_embeddings(use_sg=1):
    for bucketnum in range(4):
        class MyCorpus(object):
            """An interator that yields sentences (lists of str)."""

            def __iter__(self):
                corpus_path = f'csvs-parsed/bucket{bucketnum}.txt'
                for line in open(corpus_path, encoding='utf8'):
                    # assume there's one document per line, tokens separated by whitespace
                    yield utils.simple_preprocess(line)

        # with open(f"csvs-parsed/bucket{3}.txt", "r") as bucket:
        # 	sentences = bucket.readlines()
        sentences = MyCorpus()

        model = Word2Vec(sentences=sentences, sg=use_sg)

        with open(f"model-pickles/bucket{'sg' if use_sg else 'cbow'}{bucketnum}.pickle", mode='wb') as out:
            pickle.dump(model, out)


if __name__=='__main__':
    form_embeddings()
