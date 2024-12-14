from sklearn.ensemble import GradientBoostingRegressor

def train_meta_model(meta_X, meta_y):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(meta_X, meta_y)
    return model
