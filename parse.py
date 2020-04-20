import pandas as pd
import string
import pickle
import glob
from collections import defaultdict

test = False

def parse_blogtext():
	"""
	Method for splitting the blog posts up by age groups.
	Stores the results to text files and to a pickle file for future use.
	"""

	extension = 'csv'
	all_filenames = [i for i in glob.glob('csv/*.{}'.format(extension))]
	[print(fn) for fn in all_filenames]

	#combine the same-headered blog csv files
	df = pd.concat([pd.read_csv(f) for f in all_filenames], ignore_index=True)
	
	age_buckets = defaultdict(list)

	for index, row in df.iterrows():
		if row.age < 18:
			if test:
				row = string.replace('fish', 'buldswag')
			age_buckets[0].append(row)
		elif row.age < 30:
			age_buckets[1].append(row)
		elif row.age < 50:
			age_buckets[2].append(row)
		else:
			if test:
				row = string.replace('map', 'buldswag')
			age_buckets[3].append(row)
			print(row.age)

	with open(f"buckets.pickle", 'wb') as out:
		pickle.dump(age_buckets, out)
	for bucketnum in range(len(age_buckets)):
		print("storing age bucket ", bucketnum, "to text file")
		with open(f"csvs-parsed/bucket{bucketnum}.txt", 'w') as bucketfile:
			for row in age_buckets[bucketnum]:
				bucketfile.write(row.text.strip() + "\n")
	

if __name__ == '__main__':
	parse_blogtext()