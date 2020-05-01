from math import log10
from pycipher import SimpleSubstitution
class ngram_score(object):
    def __init__(self,ngramfile,sep=' '):
        ''' load a file containing ngrams and counts, calculate log probabilities '''
        self.ngrams = {}
        for line in open(ngramfile):
            key,count = line.split(sep) 
            self.ngrams[key] = int(count)
        self.L = len(key)
        self.N = sum(self.ngrams.values())
        #calculate log probabilities
        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key])/self.N)
        self.floor = log10(0.01/self.N)

    def score(self,texts):
        ''' compute the score(fitness) of text '''
        sumscore = 0
        words = texts.split(" ")

        ngrams = self.ngrams.__getitem__
        for text in words:
            score = 0
            for i in range(len(text)-self.L+1):
                if text[i:i+self.L] in self.ngrams: score += ngrams(text[i:i+self.L])
                else: score += self.floor
            sumscore += score
        return sumscore


ss = SimpleSubstitution()
a=ss.encipher("hello there, my name is thinh")
print(a)
