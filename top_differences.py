import pickle
from sklearn.metrics.pairwise import cosine_similarity

def get_top_differences(embeddingindex1, embeddingindex2, use_sg=1, word_to_find="exam"):

    if embeddingindex2<embeddingindex1:
        strstring = f"{embeddingindex1}{embeddingindex2}"
        with open(f"aligned-embedding-pickles/embeddings-aligned{'sg' if use_sg==1 else 'cbow'}{strstring}.pickle", 'rb') as filein:
	        secondmodel, firstmodel = pickle.load(filein)
        
    else:
        strstring = f"{embeddingindex1}{embeddingindex2}"
        with open(f"aligned-embedding-pickles/embeddings-aligned{'sg' if use_sg==1 else 'cbow'}{strstring}.pickle", 'rb') as filein:
	        firstmodel, secondmodel = pickle.load(filein)

    m1wv = firstmodel.wv
    m2wv = secondmodel.wv

        
    vocab1 = set(m1wv.index2word)
    vocab2 = set(m2wv.index2word)

    print(len(vocab1))
    print(len(vocab2))

    vocabinboth = vocab1 & vocab2



    cosinesims = dict()
    for word in vocabinboth:
        if (m1wv.vocab[word].count>50 and m2wv.vocab[word].count>50
            and len(word)>3):
            vec1 = (firstmodel[word]).reshape(1, -1)
            vec2 = (secondmodel[word]).reshape(1, -1)
            cosinesims[word] = cosine_similarity(vec1, vec2)

    sorteddifferences = sorted(cosinesims.items(), key=lambda x: x[1])

    for cosinesim in sorteddifferences[0:20]:
        print("-----------------------------")
        print(cosinesim)
        print()
        print("Most similar for younger")
        print(m1wv.most_similar([cosinesim[0]]))
        print("Most similar for older")
        print(m2wv.most_similar([cosinesim[0]]))
        print("-----------------------------")

def get_differences_for_word(embeddingindex1, embeddingindex2, use_sg=1, word_to_find="exam"):

    if embeddingindex2<embeddingindex1:
        strstring = f"{embeddingindex1}{embeddingindex2}"
        with open(f"aligned-embedding-pickles/embeddings-aligned{'sg' if use_sg==1 else 'cbow'}{strstring}.pickle", 'rb') as filein:
	        secondmodel, firstmodel = pickle.load(filein)
        
    else:
        strstring = f"{embeddingindex1}{embeddingindex2}"
        with open(f"aligned-embedding-pickles/embeddings-aligned{'sg' if use_sg==1 else 'cbow'}{strstring}.pickle", 'rb') as filein:
	        firstmodel, secondmodel = pickle.load(filein)

    m1wv = firstmodel.wv
    m2wv = secondmodel.wv

        
    vocab1 = set(m1wv.index2word)
    vocab2 = set(m2wv.index2word)

    print(len(vocab1))
    print(len(vocab2))

    vocabinboth = vocab1 & vocab2



    cosinesims = dict()
    for word in vocabinboth:
        if (m1wv.vocab[word].count>50 and m2wv.vocab[word].count>50
            and len(word)>3):
            vec1 = (firstmodel[word]).reshape(1, -1)
            vec2 = (secondmodel[word]).reshape(1, -1)
            cosinesims[word] = cosine_similarity(vec1, vec2)

    sorteddifferences = sorted(cosinesims.items(), key=lambda x: x[1])

    # for cosinesim in sorteddifferences[0:20]:
    #     print("-----------------------------")
    #     print(cosinesim)
    #     print()
    #     print("Most similar for younger")
    #     print(m1wv.most_similar([cosinesim[0]]))
    #     print("Most similar for older")
    #     print(m2wv.most_similar([cosinesim[0]]))
    #     print("-----------------------------")

    # cosinesim = cosinesims['thesis']
    # print(cosinesim)
    print(f"Most similar for younger group, group {embeddingindex1}")
    print(m1wv.most_similar([word_to_find]))
    print("Most similar for older group, group ")
    print(m2wv.most_similar([word_to_find]))
    print("-----------------------------")