from parse import parse_blogtext
from form_embeddings import form_embeddings
from procrustes_align import smart_procrustes_align_gensim
from align_embeddings import align_embeddings
from top_differences import get_top_differences, get_differences_for_word
from itertools import combinations

use_sg=1

should_parse_blogtext = True
should_form_embeddings = True
should_align_embeddings = True
should_find_top_cosine_differences = True
should_find_differences_for_word = True

first_age_partition = 0
second_age_partition = 3

if should_parse_blogtext:
    parse_blogtext()

if should_form_embeddings:
    form_embeddings(use_sg=use_sg)

if should_align_embeddings:
    for firstindex, secondindex in combinations(range(4), 2):
        align_embeddings(firstindex, secondindex, use_sg=use_sg)

if should_find_top_cosine_differences:
    get_top_differences(first_age_partition, second_age_partition, use_sg)

if should_find_differences_for_word:
    word = input("What word would you like to find differences for?")
    get_differences_for_word(first_age_partition, second_age_partition, use_sg=use_sg, word_to_find=word)
