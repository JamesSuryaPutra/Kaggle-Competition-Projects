# Petals to the Metal - Flower Classification on TPU
![header](https://github.com/JamesSuryaPutra/Petals-to-the-Metal-Flower-Classification-on-TPU/assets/155945814/1c7b215b-1ee8-43df-8269-5050e52aed69)

# Learn how to use Tensor Processing Units (TPUs) on Kaggle
TPUs are powerful hardware accelerators specialized in deep learning tasks. They were developed (and first used) by Google to process large image databases, such as extracting all the text from Street View. This competition is designed for you to give TPUs a try.

TPU quotas are available on Kaggle at no cost to users.

# The challenge
It’s difficult to fathom just how vast and diverse our natural world is. There are over 5,000 species of mammals, 10,000 species of birds, 30,000 species of fish; and astonishingly, over 400,000 different types of flowers.

In this competition, you’re challenged to build a machine learning model that identifies the type of flowers in a dataset of images (for simplicity, we’re sticking to just over 100 types).

# Evaluation
Submissions are evaluated on macro F1 score.

F1 is calculated as follows:

![f1_score](https://github.com/JamesSuryaPutra/Petals-to-the-Metal-Flower-Classification-on-TPU/assets/155945814/09814327-50db-4d7d-a790-79534f6bec83)


where:

![precision-recall](https://github.com/JamesSuryaPutra/Petals-to-the-Metal-Flower-Classification-on-TPU/assets/155945814/e4d86160-b53b-4358-b792-54434fcdc574)


In "macro" F1 a separate F1 score is calculated for each class / label and then averaged.
