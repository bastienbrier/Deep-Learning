from random import randint
import numpy as np
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

import numpy as np
from gensim.models import word2vec


def avg_word2vec(model, dataset='data/snli.test'):
    array_sentences = []
    array_embeddings = []
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            wordcount = 0 # modification
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.vocab:
                    if avgword2vec is None:
                        avgword2vec = model[word]
                    else:
                        avgword2vec = avgword2vec + model[word]
                    wordcount += 1
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                avgword2vec = avgword2vec / wordcount  # normalize sum and not len(avgword2vec)
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings


def cosine_similarity(a, b):
    assert len(a) == len(b), 'vectors need to have the same size'
    cos_sim = -1
    #########
    # TO DO : IMPLEMENT THE COSINE SIMILARITY BETWEEN a AND b
    #########
    a_b = np.dot(a, np.transpose(b))
    a_a = np.dot(a, np.transpose(a))
    b_b = np.dot(b, np.transpose(b))
    cos_sim = a_b / (np.sqrt(a_a) * np.sqrt(b_b))
    return cos_sim


def most_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])
    closest_idx = max(list_scores, key=list_scores.get)
    #########
    # TO DO : output the 5 most similar sentences
    #########
    print 'The query :\n'
    print query_sentence + '\n'
    print 'is most similar to\n'
    print array_sentences[closest_idx]
    print 'with a score of : {0}'.format(list_scores[closest_idx])

    return closest_idx

def most_5_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])
    #########
    # TO DO : find and output the 5 most similar sentences
    #########
    closest_5_idx = []
    closest_5_scores = []
    for i in range(5):
        closest_idx = max(list_scores, key=list_scores.get)
        closest_5_idx.append(closest_idx)
        closest_5_scores.append(list_scores[closest_idx])
        list_scores[closest_idx] = 0
        closest_idx = 0

    return (closest_5_idx, closest_5_scores)


def IDF(dataset='data/snli.test'):
    # Compute IDF (Inverse Document Frequency). Here a "document" is a sentence.
    # word2idf['peach'] = IDF(peach)
    wordcount = {}
    word2idf = {}
    count_docs = 0
    with open(dataset) as f:
        for line in f:
            sentence = line.split()
            count_docs += 1
            for word in sentence:
                if word in wordcount: # if in the dictionary, increment by 1
                    wordcount[word] += 1
                else: # else add it
                    wordcount[word] = 1

    for key in wordcount:
        word2idf[key] = np.log(float(count_docs) / float(wordcount[key]))
    return word2idf

def avg_word2vec_idf(model, word2idf, dataset='data/snli.test'):
    # TODO : Modify this to have a weighted (idf weights) average of the word vectors
    """

    :type model: object
    """
    array_sentences = []
    array_embeddings = []
    weights_sum = 0
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.vocab:
                    if avgword2vec is None:
                        # TODO : ADD WEIGHTS
                        avgword2vec = word2idf[word] * model[word]
                        weights_sum = word2idf[word]
                    else:
                        # TODO : ADD WEIGHTS
                        avgword2vec = avgword2vec + word2idf[word] * model[word]
                        weights_sum += word2idf[word]
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                # TODO : NORMALIZE BY THE SUM OF THE WEIGHTS
                avgword2vec = avgword2vec / weights_sum  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings

if __name__ == "__main__":

    if False: # FIRST PART
        sentences = word2vec.Text8Corpus('data/text8')

        # Train a word2vec model
        embedding_size = 200
        your_model = word2vec.Word2Vec(sentences, size=embedding_size)
        #########
        # TO DO : Report from INFO :
            # - total number of raw words found in the corpus.
            # - number of words retained in the vocabulary (with min_count = 5)
        #########

        # Train a word2vec model with phrases
        # bigram_transformer = gensim.models.Phrases(sentences)
        # your_model_phrase = Word2Vec(bigram_transformer[sentences], size=200)

    if True: # SECOND PART

        """
        Investigating word2vec word embeddings space
        """
        # Loading model trained on words
        model = word2vec.Word2Vec.load('models/text8.model')

        # Loading model enhanced with phrases (2-grams)
        model_phrase = word2vec.Word2Vec.load('models/text8.phrase.model')

        # Words that are similar are close in the sense of the cosine similarity.
        sim = model.similarity('woman', 'man')
        sim_2 = model.similarity('apple', 'mac')
        sim_3 = model.similarity('apple', 'peach')
        sim_4 = model.similarity('banana', 'peach')
        print 'Printing word similarity between "woman" and "man" : {0}'.format(sim)
        print 'Printing word similarity between "apple" and "mac" : {0}'.format(sim_2)
        print 'Printing word similarity between "apple" and "peach" : {0}'.format(sim_3)
        print 'Printing word similarity between "banana" and "peach" : {0}'.format(sim_4)

        # And words that appear in the same context have similar word embeddings.
        sim_paris = model.most_similar(['paris'])
        sim_ph_paris = model_phrase.most_similar(['paris'])
        print 'Printing model similar words to "paris" :{0}'.format(sim_paris)
        print 'Printing model_phrase similar words to "paris" :{0}'.format(sim_ph_paris)
        sim_diff = model.most_similar(['difficult'])
        sim_ph_diff = model_phrase.most_similar(['difficult'])
        print 'Printing model similar words to "difficult" :{0}'.format(sim_diff)
        print 'Printing model_phrase similar words to "difficult" :{0}'.format(sim_ph_diff)
        sim_ph_clint = model_phrase.most_similar(['clinton'])
        print 'Printing model_phrase similar words to "clinton" :{0}'.format(sim_ph_clint)

        # Compositionality and structure in word2vec space
        sim_vect = model.most_similar(positive=['woman', 'king'], negative=['man'])
        print 'Printing closest word to "vect(woman) - vect(man) + vect(king)" :{0}'.format(sim_vect)

        #########
        # TO DO : Compute similarity (france, berlin, germany)
        #########
        sim_vect_fr = model.most_similar(positive=['france', 'berlin'], negative=['germany'])
        print 'Printing closest word to "vect(france) - vect(germany) + vect(berlin)" :{0}'.format(sim_vect_fr)

        # Explore the space
        sim_roll = model_phrase.most_similar(['rolling_stones'])
        print 'Printing model similar words to "rolling_stones" :{0}'.format(sim_roll)
        sim_vect_nap = model.most_similar(positive=['germany', 'napoleon'], negative=['france'])
        print 'Printing closest word to "vect(germany) - vect(france) + vect(napoleon)" :{0}'.format(sim_vect_nap)
        sim_cher = model_phrase.most_similar(['hollywood'])
        print 'Printing model similar words to "hollywood" :{0}'.format(sim_cher)
        sim_holly = model_phrase.similarity('hollywood', 'los_angeles')
        print 'Printing word similarity between "hollywood" and "los_angeles" : {0}'.format(sim_holly)

    if True: # THIRD PART
        """
        Sentence embeddings with average(word2vec)
        """
        data_path = 'data/snli.test'
        array_sentences, array_embeddings = avg_word2vec(model, dataset=data_path)

        #########
        # TO DO : do the TODOs in cosine_similarity
        #########
        query_idx =  777 # random sentence
        assert query_idx < len(array_sentences) # little check

        # For the next line to work, you need to implement the "cosine_similarity" function.
        # array_sentences[closest_idx] will be the closest sentence to array_sentences[query_idx].
        closest_idx = most_similar(query_idx, array_embeddings, array_sentences)

        #########
        # TO DO : Implement the most_5_similar function to output the 5 sentences that are closest to the query.
        # TO DO : Report the 5 most similar sentences to query_idx = 777
        #########
        (closest_5_idx, closest_5_scores) = most_5_similar(query_idx, array_embeddings, array_sentences)

        count = 0
        for idx in closest_5_idx:
            #########
            # TO DO: Print the 5 most similar sentences to query_idx using closest_5_idx, array_sentences, array_embeddings
            #########
            print 'The query : ' + array_sentences[query_idx] + ' is most similar to : ' + array_sentences[idx] + 'with a score of : {0}'.format(closest_5_scores[count]) + '\n'
            count += 1

    if True: # FOURTH PART
        #######
        # Weighted average of word vectors with IDF.
        #######
        word2idf = IDF(data_path)
        print 'IDF the: ' + str(word2idf["the"])
        print 'IDF a: ' + str(word2idf["a"])
        #print 'IDF clinton: ' + str(word2idf["clinton"])
        array_sentences_idf, array_embeddings_idf = avg_word2vec_idf(model, word2idf, dataset=data_path)
        closest_idx_idf = most_similar(query_idx, array_embeddings_idf, array_sentences_idf)