def tfidf_and_countvec(df, c):
    df[c] = df[c].astype(str)
    df[c].fillna('-1', inplace=True)
    group_df = df.groupby(['uid']).apply(lambda x: x[c].tolist()).reset_index()
    group_df.columns = ['uid', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_tv = TfidfVectorizer()
    enc_cv = CountVectorizer()
    tfidf_vec = enc_tv.fit_transform(group_df['list'])
    count_vec = enc_cv.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2)
    tv_svd = svd_enc.fit_transform(tfidf_vec)
    cv_svd = svd_enc.fit_transform(count_vec)
    tv_svd = pd.DataFrame(tv_svd)
    cv_svd = pd.DataFrame(cv_svd)
    tv_svd.columns = ['svd_tfidf_{}_{}'.format(c, i) for i in range(10)]
    cv_svd.columns = ['svd_countvec_{}_{}'.format(c, i) for i in range(10)]
    group_df = pd.concat([group_df, tv_svd, cv_svd], axis=1)
    del group_df['list']
    return group_df


def word2vec(df,c):
    seq = df[['uid',c]].groupby('uid').agg({c: lambda x:x.tolist()})
    input = seq.values
    sequence=[]
    for i in range(input.shape[0]):
        sequence.append([str(i) for i in list(input[i][0])])
    model= Word2Vec(sequence, size=10 , window=10 ,min_count=1, seed=2, workers=6, sg=1, iter=10)
    vectors = model.wv.vectors
    words = [word for word in model.wv.index2word]
    vector_list = []
    for i in range(10):
        vector_list.append(c+'_emb_{}'.format(i+1))
    res = pd.DataFrame()
    res[c] = words
    for i in range(10):
        res[vector_list[i]] = vectors[:,i].astype(np.float32)
    return res
