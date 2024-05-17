from sklearn.decomposition import NMF

def pca_pretrain(X, component = 20):
    pca = NMF(n_components=component)
    X_new = pca.fit_transform(X)
    return X_new