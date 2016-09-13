# -*- coding: utf-8 -*-
import string
import nltk.stem.snowball
class MoTextPreprocessor:
    exclude = set(string.punctuation)
    @classmethod
    def normalize(cls,s,language="en", removeUmlaut = True, lowerCase = True, stemming =False, removePunc = True):
        s = s.strip()
        if removePunc:
            s = ''.join(ch for ch in s if ch not in cls.exclude)
        if lowerCase:
            s = s.lower()
        if (language =="de"):
            if removeUmlaut:
                s = s.replace(u'Ä','ae')
                s = s.replace(u'ä','ae')
                s = s.replace(u'Ü','ue')
                s = s.replace(u'ü','ue')
                s = s.replace(u'Ö','oe')
                s = s.replace(u'ö','oe')
                s = s.replace(u'ß','ss')
            if stemming:
                stm = nltk.stem.snowball.GermanStemmer()
                s = ' '.join(stm.stem(ch) for ch in s.split())    
        if (language == "en"):
            if stemming:
                stm = nltk.stem.snowball.EnglishStemmer()
                s = ' '.join(stm.stem(ch) for ch in s.split())

        return s
# nltk.download()

# print stm.stem()
# preprocessingConfig = dict(language = "en", stemming=True,removeUmlaut = False, lowerCase = True,removePunc=True)
# print MoTextPreprocessor.normalize(u"information hasnt been detailed enough",**preprocessingConfig)
