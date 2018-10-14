from models import ProjectedAttentionTextRNN, TextAuthorRNN
from preprocess import generate_data
from utils import train
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss


def run_text():
    data = generate_data(size=200000, split_point=160000, emb_size=50, max_len=25)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    test_batches = data['test_batches']
    model = ProjectedAttentionTextRNN(emb_matrix)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    train(model, train_batches, test_batches, optimizer, criterion, 50, 5)


def run_text_author():
    data = generate_data(size=1000000, split_point=960000, emb_size=25, max_len=25)
    a2i = data['a2i']
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    test_batches = data['test_batches']
    model = TextAuthorRNN(emb_matrix, len(a2i))
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    train(model, train_batches, test_batches, optimizer, criterion, 50, 5)

if __name__ == "__main__":
    run_text_author()
