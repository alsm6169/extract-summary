import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# text = ["This is 000 Sparta!"]
text = ['We expect the market to grow by 30% CAGR to reach USD 90bn by 2025.',
        'As a result, we believe conversational AI’s share in the broader AI’s addressable market '
        'can climb to 20% by 2025 (USD 18–20bn).']
tfIdfVectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z0-9]+\\w*\\b', use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(text)
feature_names = tfIdfVectorizer.get_feature_names_out()
print(feature_names.shape, feature_names)
idx = 0
print(idx, tfIdf[idx].T.toarray().flatten(order='C').shape, '\n', tfIdf[idx].T.toarray().flatten(order='C'))
idx = 1
print(idx, tfIdf[idx].T.toarray().flatten(order='C').shape, '\n', tfIdf[idx].T.toarray().flatten(order='C'))

# df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
# df = df.sort_values('TF-IDF', ascending=False)
# print(df.head(25))