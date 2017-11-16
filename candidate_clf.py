import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

shop = pd.read_csv('shop_info.csv')
mall_list = shop['mall_id'].unique()
for ii,mall in enumerate(mall_list):
	print(ii,'mall start')
	data = pd.read_csv('proba_file/mb_xgb_0_minute/'+mall+'t.csv')
	col = data.columns
	data = data.values
	K = 5
	ans = []
	for line in data:
		sorted_line = sorted(line,reverse=True)
		i = 0
		ans_line = []
		for v in sorted_line:
			ans_line.append(col[np.where(line==v)[0]][0])
			i += 1
			if i >= K:
				break

		ans.append(ans_line)
	new_data = pd.DataFrame(ans)
	new_data.to_csv('candidate/'+mall+'.csv',index=False)
