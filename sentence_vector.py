from time import time
import random
import gensim


def read_corpus(fname, tokens_only=False):
    with open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


def get_vector_from_sentence(sentence):
    # model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
    model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
    return model.infer_vector(gensim.utils.simple_preprocess(sentence))


def train_model():
    train_corpus = list(read_corpus('./data/messages_train.txt'))
    test_corpus = list(read_corpus('./data/messages_test.txt', tokens_only=True))

    print(train_corpus[:2])

    try:
        model = gensim.models.Doc2Vec.load('doc2vec.model')
    except Exception:
        print('Training new model')
        start = time()
        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print(time() - start, 'seconds to train model')
        model.save('doc2vec.model')


# Pick a random document from the test corpus and infer a vector from the model
    doc_id = random.randint(0, len(test_corpus) - 1)
    inferred_vector = model.infer_vector(test_corpus[doc_id])
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
    print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
