import collections
import re
import random
import matplotlib.pyplot as plt
# from pycipher import SimpleSubstitution
import sys


###############################################################
## program paramaters

# Path to the text file contaning the ciphertext
CIPHERTEXT = sys.argv[2]

# Path to the text file contaning the plaintext
PLAINTEXT = "plaintext.txt"

# Path to the text file contaning the bigram, trigram, quadgram of english
PATH_NGRAM = {  1: "english_monograms.txt",
                2: "english_bigrams.txt",
                3: "english_trigrams.txt",
                4: "english_quadgrams.txt"}

# Encrypted chars in the ciphertext
CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# Size of the population to use for the genetic algorithm
# POPULATION_SIZE = 50
POPULATION_SIZE = 20

# Size of the population slice of best peforming solutions to keep at each
# iteration
TOP_POPULATION = 10

# Number of intervals for which the best score has to be stable before aborting
# the genetic algorithm
STABILITY_INTERVALS = 20

# Number of crossovers to execute for each new child in the genetic algorithm
CROSSOVER_COUNT = 2

# Number of random mutation to introduce for each new child in the genetic
# algorithm
MUTATIONS_COUNT = 1

#################################################################
# Implememtation

# converting to ngram
def ngramWord(word,n_gram ):
    """
    :param: word: any word, n_gram: [2,3,4] extract ngram feature
    :return: list of ngram feature
    eg.. ngramWord("hello",2) -> [he,el,ll,lo]
    """
    n_wise = [word[i:i + n_gram] for i in range(0, len(word) - (n_gram-1), 1)]
    return n_wise

def ngrams(text, n_gram):
    # extract ngrams from text
    counter = collections.Counter()
    words = re.sub('[^A-Za-z]', ' ', text).split() #replace all non alphabet by blank, then split()
    for word in words:
        for n_wise in ngramWord(word,n_gram):
            counter[n_wise] += 1
    return counter


def decode(ciphertext, key):
    cleartext = ''
    for char in ciphertext:
        cleartext += key.get(char, char)
    return cleartext


def init_mapping():
    # Generate a randomly initialized solution
    # {'A': 'U', 'U': 'A', 'B': 'F', 'F': 'B', 'C': 'R', 'R': 'C', 'D': 'P', 'P': 'D', 'E': 'W', 'W': 'E', 'G': 'H', 'H': 'G', 'I': 'M', 'M': 'I', 'J': 'Q', 'Q': 'J'...}
    repls = set(CHARS)
    mapping = {}
    for c in CHARS:
        if c in mapping:
            continue
        repl = random.choice(list(repls))
        repls.remove(repl)
        repls.discard(c) #remove from the set if it's present, maybe c == repl
        mapping[c] = repl
        mapping[repl] = c
    return mapping


def update_mapping(mapping, char, repl):
    # Update the solution by switching `char` with `repl`
    # and `repl` with `char`.
    current_repl = mapping[char]
    current_char = mapping[repl]

    if current_char == repl:
        current_char = current_repl
    elif current_repl == char:
        current_repl = current_char

    mapping[current_char] = current_repl
    mapping[current_repl] = current_char

    mapping[char] = repl
    mapping[repl] = char


############################################
# Genetic algorithm routines
from math import log10
def score(text, ref_ngrams, n_gram ):
    """
    compute the score "fitness" of text
    score = number of ngram occurrences in text multip number of ngram occur in english
    """
    N = sum(ref_ngrams.values())
    text_ngram = ngrams(text, n_gram)
    score = 0
    for n_wise, occurrences in text_ngram.items():
        score += occurrences * ref_ngrams[n_wise]
    return score/N


def select(population, ciphertext, ref_ngrams , n_gram):
    scores = []

    # Compute the score of each solution
    for p in population:
        scores.append((score(decode(ciphertext, p), ref_ngrams, n_gram), p))

    # Sort the solutions by their score
    sorted_population = sorted(scores, reverse=True)

    # Select only the best TOP_POPULATION solutions
    selected_population = sorted_population[:TOP_POPULATION]

    return selected_population[0][0], [m for _, m in selected_population]

def generate(population):
    new_population = population[:]
    while len(new_population) < POPULATION_SIZE:
        # Randomly select two parent solutions
        x, y = random.choice(population), random.choice(population)

        # Create the child solution
        child = x.copy()

        # Switch CROSSOVER_COUNT chromosomes between the parents
        for i in range(CROSSOVER_COUNT):
            char = random.choice(list(CHARS))
            update_mapping(child, char, y[char])

        # Randomly mutate MUTATIONS_COUNT chromosomes of the the child solution
        for i in range(MUTATIONS_COUNT):
            char = random.choice(list(CHARS))
            repl = random.choice(list(CHARS))
            update_mapping(child, char, repl)

        # Add the newly obtained child the the current population
        new_population.append(child)
    return new_population

###############################################################################
# Decryption routine


def decrypt(n_gram, plot_color):
    BestScores = []
    Iterations = []
    ref_ngram = {}
    for line in open(PATH_NGRAM[n_gram]):
        key, count = line.split(" ")
        ref_ngram[key] = int(count)

    with open(CIPHERTEXT, encoding="utf8") as fh:
        ciphertext = fh.read().upper()

    # Create an initial population of random possible solutions
    population = [init_mapping() for i in range(POPULATION_SIZE)]
    print("population 0: ", population[0], sep=" ")

    # Set the initial values for the stability checker
    last_score = 0
    last_score_increase = 0
    iterations = 0

    # Run the genetic algorithm
    while last_score_increase < STABILITY_INTERVALS:
        # Fill up the population up to POPULATION_SIZE solutions by crossing
        # over and mutating the TOP_POPULATION best solutions
        population = generate(population)

        # Select the TOP_POPULATION best solutions from the current population
        best_score, population = select(population, ciphertext, ref_ngram, n_gram)

        # Update the stability check state with the current best score
        if best_score > last_score:
            last_score_increase = 0
            last_score = best_score
        else:
            last_score_increase += 1
        print(iterations ,":", last_score_increase, ':', best_score)
        BestScores.append(best_score)
        Iterations.append(iterations)
        plt.plot(Iterations, BestScores, plot_color)
        plt.xlabel('generation')
        plt.ylabel('bestScore')
        plt.show(block=False)
        plt.pause(0.05)
        # plt.close()
        iterations += 1


    print('Best solution found after {} iterations'.format(iterations))
    print('with population :{}'.format(population[0]))

    return decode(ciphertext, population[0]), population[0],BestScores[-1]

def metrics(n_gram, number_iter_GA = 3):
    colorMapping = {1:"b",
                    2:"g",
                    3:"r",
                    4:"c",
                    5:"m",
                    6:"y",
                    7:"k",
                    8:"w"}

    f = open("predPlaintext.txt","w", encoding="utf8")
    for i in range(number_iter_GA):

        AnswerText, key, bestScores= decrypt(n_gram,colorMapping[i+1])
        f.write("------------------------------------------------------------------------------------------------------\n")
        f.write("iteration {}\n".format(i+1))
        f.write(str(key)+"\n")
        f.write("Best Score: {}\n".format(bestScores))
        f.write("------------------------------------------------------------------------------------------------------\n")
        f.write(AnswerText.lower())
        f.write("\n\n\n")
    f.close()
    # totalLen = len(AnswerText)
    # accuracyCount = 0
    # with open(PLAINTEXT,encoding="utf8") as original:
    #     OriginaText = original.read().upper()
    # for i in range(totalLen):
    #     if OriginaText[i] == AnswerText[i]:
    #         accuracyCount += 1
    # print('accuracy : {}%'.format(100 * accuracyCount / totalLen))

# plt.plot(BestScores, Iterations)
if len(sys.argv) == 3:
    metrics(int(sys.argv[1])) # ngram = 2,3,4
if len(sys.argv) >3:
    metrics(int(sys.argv[1]),int(sys.argv[3]))
plt.show()
plt.pause(2)
plt.close()


