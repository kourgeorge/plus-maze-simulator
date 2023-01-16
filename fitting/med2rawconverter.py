import numpy as np
import pandas as pd


def load_and_parse_med_file(file_path):
	data = pd.read_csv(file_path, skiprows=40, header=None, delimiter=r"\s+")
	data.drop([0], axis=1, inplace=True)
	l = data.values.tolist()
	flat_list = [item for sublist in l for item in sublist]
	my_lists = list(zip(*[iter(flat_list)] * 10))

	df = pd.DataFrame(my_lists, columns=['trial', 'correct_p1', 'correct_p2', 'zeros', 'chosen_arm', 'reaction_time',
										 'arm_response_time', 'trial_outcome', 'configuration', 'zeros2'])

	df.drop(['zeros', 'zeros2'], axis=1, inplace=True)
	df = df.dropna()
	for column in ['trial', 'correct_p1', 'correct_p2','chosen_arm', 'trial_outcome', 'configuration']:
		df[column] = df[column].apply(lambda x: float(x))
		df[column] = df[column].apply(lambda x: int(x) if not np.isnan(x) else x)
		df[column] = df[column].astype(int, errors='ignore')

	column_with_zeros = df.loc[df['trial'] == 0].index
	if len(column_with_zeros)>0:
		df = df.head(column_with_zeros[0])
	return df


if __name__ == '__main__':
	df = load_and_parse_med_file('/Users/gkour/repositories/plusmaze/fitting/med_data/!2017-10-31_11h05m.Subject 1')
