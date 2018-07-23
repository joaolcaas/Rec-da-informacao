from surprise import SVD
from surprise import Dataset
from surprise import KNNBasic

'''
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = KNNBasic()
algo.fit(trainset)

uid = str(196)
iid = str(302)
pred = algo.predict(uid,iid,r_ui = 4,verbose = True)
'''
from collections import defaultdict

from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader

import os

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First train an SVD algorithm on the custom dataset.
# Pode botar tb: timestamp
reader = Reader(line_format='user item rating', sep=' ', skip_lines=3, rating_scale=(1, 5))

#custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) + '/custom_dataset')
#print("> Using: " + custom_dataset_path)
print("> Loading data...")
data = Dataset.load_builtin('ml-100k')
print("> OK")

print("> Creating trainset...")
trainset = data.build_full_trainset()
print("> OK")

print("> Training...")
algo = KNNBasic()
algo.train(trainset)
print("> OK")

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
print("> Predicting...")
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
print("> OK")

# Print the recommended items for each user
print("> Results:")
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
