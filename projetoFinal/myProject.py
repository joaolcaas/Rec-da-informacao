from surprise import SVD
from surprise import Dataset
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split
from collections import defaultdict

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=.15)

algo = KNNBasic()

test_user = str(543)

algo.fit(trainset)

predictions = algo.test(testset)

accuracy.rmse(predictions)

top_n = defaultdict(list)

for uid, iid, true_r, est, _ in predictions:
	top_n[uid].append((iid, est))


print(top_n[test_user])
sorted_by_second = sorted(top_n[test_user][0], key=lambda x:x[1])
'''
for i in top_n[test_user]:		
	sorted(i, key=lambda x: x[1])
	print(i)
'''
