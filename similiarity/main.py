from itertools import product

from scipy.stats import spearmanr
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

with open("Task_4_sample_8.txt", encoding="utf-8") as rf:
  triples = [line.strip().split(",") for line in rf.readlines()]
  score_map = {tuple(triple[:2]): float(triple[2]) for triple in triples}

for w1, w2 in list(score_map)[:2]:

  ss1 = wn.synset(w1 + ".n.01")
  ss2 = wn.synset(w2 + ".n.01")

  print("\nPath: %.3f" % ss1.path_similarity(ss2), end=" ")
  print("\nwup: %.3f" % ss1.wup_similarity(ss2), end=" ")
  print("\nshortest_path: %.3f" % ss1.shortest_path_distance(ss2))

list_pairs = list(score_map)
wup_list, true_list, path_list = [], [], []
lch_list = []

# для всех пар
for w1, w2 in list_pairs:

  try:
    all_w1 = wn.synsets(w1, pos="n")
    all_w2 = wn.synsets(w2, pos="n")

    # we add metrics of interest and expert reviews
    wups = [item1.wup_similarity(item2) for item1, item2 in product(all_w1, all_w2)]
    wup = max(wups) if len(wups)>0 else 0

    lchs = [item1.lch_similarity(item2) for item1, item2 in product(all_w1, all_w2)]
    lch = max(lchs) if len(lchs)>0 else 0


    paths = [item1.path_similarity(item2) for item1, item2 in product(all_w1, all_w2)]
    path = max(paths) if len(paths)>0 else 0

    if(len(wups)==0 or len(lchs)==0 or len(paths)==0):
      print(w1, w2, "misunderstanding")
      print(wups, lchs, paths)
      continue

    wup_list.append(wup)
    path_list.append(path)
    lch_list.append(lch)

    true_list.append(score_map[(w1, w2)])

  except Exception as e:
    print(w1, w2, "error:", e)

# spearman coef is a kind of mse metric but it is a kind of normalized (1-6*[sum((true-real)^2)]/[n*(n^2 -1)])
# so 1 shows the ideal predictions
coef, p = spearmanr(wup_list, true_list)
print("wup  Spearman R: %.4f" % coef)

coef, p = spearmanr(path_list, true_list)
print("path Spearman R: %.4f" % coef)

coef, p = spearmanr(lch_list, true_list)
print("lch Spearman R: %.4f" % coef)

w = wn.synset('boy.n.01')
hyponyms_list = w.hyponyms()
print(len(hyponyms_list))
sorted_hyponyms = sorted(hyponyms_list, key=lambda h: h.name())
if sorted_hyponyms:
    first_hyponym_name = sorted_hyponyms[0].name()
else:
    first_hyponym_name = None
print(first_hyponym_name)