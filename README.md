Petals to the Metal: Flower Classification on TPU

# The Challenge
It’s difficult to fathom just how vast and diverse our natural world is. There are over 5,000 species of mammals, 10,000 species of birds, 30,000 species of fish; and astonishingly, over 400,000 different types of flowers.

In this competition, you’re challenged to build a machine learning model that identifies the type of flowers in a dataset of images (for simplicity, we’re sticking to just over 100 types).

# Evaluation
Submissions are evaluated on macro F1 score.

F1 is calculated as follows:
F1 = 2 ∗ precision ∗ recall / precision + recall

where:
precision = TP / TP + FP
recall = TP / TP + FN

In "macro" F1 a separate F1 score is calculated for each class / label and then averaged.

# Submission File
For each id in the test set, you must predict a type of flower (or label). The file should contain a header and have the following format:
***************
id,label
a762df180,0
24c5cf439,0
7581e896d,0
eb4b03b29,0
etc.
***************
