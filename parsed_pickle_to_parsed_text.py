import pandas as pd
import pickle

with open("buckets.pickle", 'rb') as filein:
	buckets = pickle.load(filein)


for bucketnum in range(len(buckets)):
	print(bucketnum)
	with open(f"csvs-parsed/b2cket{bucketnum}.txt", 'w') as bucketfile:
		for row in buckets[bucketnum]:
			bucketfile.write(row.text.strip() + "\n")
	

