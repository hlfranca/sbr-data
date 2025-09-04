import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


anchor_data = pd.read_csv('explanation_logs/anchor_fold1_20240506-030237.csv')
anchor_data = anchor_data.infer_objects()
security_anchor_data = anchor_data[anchor_data['Prediction'] == 'Positive']
security_anchor_words = []
for index, sample_report in security_anchor_data.iterrows():
    temp_anchors = sample_report['Anchors'].split(' AND ')
    security_anchor_words.extend(temp_anchors)
security_anchor_words = np.array(security_anchor_words)
security_exhibition_anchor_data = pd.DataFrame(columns=['Word', 'Condition'])
for anchor_description in security_anchor_words:
    word = ''
    for segment in anchor_description.split():
        if segment.isalpha():
            word = segment
    condition = 'Absence'
    if '>' in anchor_description:
        condition = 'Presence'
    security_exhibition_anchor_data = pd.concat([security_exhibition_anchor_data if not security_exhibition_anchor_data.empty else None,
                                                 pd.DataFrame.from_dict({'Word': [word], 'Condition': [condition]})])
count_security_anchor_data = security_exhibition_anchor_data.value_counts().reset_index(name='Count')
count_security_anchor_data = count_security_anchor_data.groupby('Word', as_index=False).agg({'Count': 'sum'})
count_security_anchor_data = count_security_anchor_data.sort_values('Count', ascending=False)
count_security_anchor_data['Word'].replace('', np.nan, inplace=True)
count_security_anchor_data = count_security_anchor_data.dropna(subset='Word')
top_security_anchors = list(count_security_anchor_data['Word'].head(10))
security_exhibition_anchor_data = security_exhibition_anchor_data[security_exhibition_anchor_data['Word'].isin(top_security_anchors)]

non_security_anchor_data = anchor_data[anchor_data['Prediction'] == 'Negative']
non_security_anchor_words = []
for index, sample_report in non_security_anchor_data.iterrows():
    temp_anchors = sample_report['Anchors'].split(' AND ')
    non_security_anchor_words.extend(temp_anchors)
non_security_anchor_words = np.array(non_security_anchor_words)
non_security_exhibition_anchor_data = pd.DataFrame(columns=['Word', 'Condition'])
for anchor_description in non_security_anchor_words:
    word = ''
    for segment in anchor_description.split():
        if segment.isalpha():
            word = segment
    condition = 'Absence'
    if '>' in anchor_description:
        condition = 'Presence'
    non_security_exhibition_anchor_data = pd.concat([non_security_exhibition_anchor_data if not non_security_exhibition_anchor_data.empty else None,
                                                     pd.DataFrame.from_dict({'Word': [word], 'Condition': [condition]})])
count_non_security_anchor_data = non_security_exhibition_anchor_data.value_counts().reset_index(name='Count')
count_non_security_anchor_data = count_non_security_anchor_data.groupby('Word', as_index=False).agg({'Count': 'sum'})
count_non_security_anchor_data = count_non_security_anchor_data.sort_values('Count', ascending=False)
count_non_security_anchor_data['Word'].replace('', np.nan, inplace=True)
count_non_security_anchor_data = count_non_security_anchor_data.dropna(subset='Word')
top_non_security_anchors = list(count_non_security_anchor_data['Word'].head(10))
non_security_exhibition_anchor_data = non_security_exhibition_anchor_data[non_security_exhibition_anchor_data['Word'].isin(top_non_security_anchors)]

coefficient_data = pd.read_csv('explanation_logs/coef_fold1_20240505-223308.csv')
coefficient_data = coefficient_data.infer_objects()
coefficient_data = coefficient_data.sort_values('Coefficient', ascending=False, key=abs)

shap_data = pd.read_csv('explanation_logs/shap_fold1_20240505-223308.csv')
shap_data = shap_data.infer_objects()
shap_data = shap_data.drop('Unnamed: 0', axis=1)
shap_data = shap_data.reindex(shap_data.mean().sort_values(ascending=False, key=abs).index, axis=1)
shap_top_words = shap_data.columns[: 10]
exhibition_shap_data = pd.DataFrame(columns=['Word', 'SHAP'])
for word in shap_top_words:
    for shap_value in shap_data[word]:
        exhibition_shap_data = pd.concat([exhibition_shap_data if not exhibition_shap_data.empty else None,
                                          pd.DataFrame.from_dict({'Word': [word], 'SHAP': [shap_value]})])


all_relevant_words = pd.concat([df['Word'] for df in (coefficient_data.head(10), exhibition_shap_data)],
                               ignore_index=True).unique()
colors = sns.color_palette('Paired', len(all_relevant_words))
palette = dict(zip(all_relevant_words, colors))

conditions = ['Absence', 'Presence']
colors2 = sns.color_palette('hls', len(conditions))
palette2 = dict(zip(conditions, colors2))

fig = plt.figure(figsize=(24, 10))
plt.rc('font', size=12)

ax = fig.add_subplot(4, 1, 1)
plt.rc('font', size=12)
plt.title('a) Coefficients')
sns.barplot(data=coefficient_data.head(10), x='Word', y='Coefficient', palette=palette)
plt.xlabel('Words')
plt.tight_layout()


ax = fig.add_subplot(4, 1, 2)
plt.rc('font', size=12)
plt.title('b) Shap values')
sns.stripplot(data=exhibition_shap_data, x='Word', y='SHAP', hue='Word', legend=False, palette=palette)
plt.xlabel('Words')
plt.ylabel('SHAP Value')
plt.tight_layout()


ax = fig.add_subplot(4, 1, 3)
plt.rc('font', size=12)
plt.title('c) Security Anchors')
sns.countplot(data=security_exhibition_anchor_data, x='Word', hue='Condition', log=True, palette=palette2,
              order=security_exhibition_anchor_data['Word'].value_counts().index)
plt.xlabel('Words')
plt.ylabel('Instances')
plt.tight_layout()

ax = fig.add_subplot(4, 1, 4)
plt.rc('font', size=12)
plt.title('d) Non-Security Anchors')
sns.countplot(data=non_security_exhibition_anchor_data, x='Word', hue='Condition', log=True, palette=palette2,
              order=non_security_exhibition_anchor_data['Word'].value_counts().index)
plt.xlabel('Words')
plt.ylabel('Instances')
plt.tight_layout()

plt.show()
