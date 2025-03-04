import io
import re
ref = "NagaokaTigrinyaCorpus1_0_rom_T2.xml"
ref_out = "Text111"


def sentenceToWordList (sentence):
    sentence_str = ""
    
    for el in sentence:
        parts = el.split(">")
        word = parts[1][:len(parts[1])-3]
        pos_tag = parts[0].split("\"")[1]
        sentence_str += word + " "

    return sentence_str.strip()

print ("program start")

f = io.open(ref, mode="r", encoding="utf-8")
lines = f.readlines()
f.close()

sentence = []

all_sentences = []

for line in lines:
    line = line.strip()
    
    if (line.startswith ("<s n=")):
        sentence = []
        continue

    if (line == "</s>"):
        all_sentences.append(sentenceToWordList (sentence))
        continue

    sentence.append(line)

f = io.open(ref_out, mode="w", encoding="utf-8")
for sentence in all_sentences:
    f.write(sentence  + "\n")
f.close()

print ("program end. See file '" + ref_out + "'")
 
