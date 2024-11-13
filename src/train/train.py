from data_loader import apply_preprocess, tokenize_text

def train(train_path, test_path):
    train_df, test_df = apply_preprocess(train_path, test_path)

