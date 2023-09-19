import pandas as pd
import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
# import numpy as np
import torch

nltk.download("punkt")


# read the data
df = pd.read_csv("data/no_duplicates_dialog_acts.csv")

# tokenize the sentences
sentences_tokenized = []
for sent in df["text"]:
    sentences_tokenized.append(word_tokenize(sent))

tagged_sentences = [TaggedDocument(d, [i]) for i, d in enumerate(sentences_tokenized)]

# instantiate embedding model
VECTOR_SIZE = 20
WINDOW = 2
MIN_COUNT = 1
EPOCHS = 100
embedding_model = Doc2Vec(
    tagged_sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    epochs=EPOCHS,
)


# get training embeddings
x_embeddings = [embedding_model.infer_vector(sent) for sent in sentences_tokenized]

# convert labels to numeric values
labels = df["label"].unique()
label_to_index = {
    labels[x]: x for x in range(len(labels))
}  # create a dictionary mapping labels to index
index_to_label = {
    v: k for (k, v) in label_to_index.items()
}  # create a dictionary mapping index to labels
y = [label_to_index[label] for label in df["label"]]

# make train and test splits
x_train, x_test, y_train, y_test = train_test_split(x_embeddings, y, test_size=0.15)


# define classification model
class Regression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regression, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, 15)
        self.layer_2 = torch.nn.Linear(15, 15)
        self.layer_3 = torch.nn.Linear(15, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


# function to calculate accuracy
def calculate_accuracy(preds, targets):
    count = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            count += 1

    return count / len(preds)


# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# instatiate model: input size is length of embedding vector(15), output size is number of classes(15)
model = Regression(VECTOR_SIZE, len(labels)).to(device)

# specifying optimizer and loss function
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# make tensors
x_train = torch.stack([torch.FloatTensor(ten) for ten in x_train]).to(device)
x_test = torch.stack([torch.FloatTensor(ten) for ten in x_test]).to(device)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# the final training loop
epochs = 1000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    train_logits = model(x_train)
    outputs = [torch.argmax(item).item() for item in train_logits]

    loss = criterion(train_logits, y_train)
    loss.backward()

    optimizer.step()
    # Testing
    best_model = model
    highest_accuracy = 0

    model.eval()
    with torch.inference_mode():
        test_logits = model(x_test)
        test_outputs = [torch.argmax(item).item() for item in test_logits]
        test_loss = criterion(test_logits, y_test)
        train_accuracy = calculate_accuracy(outputs, y_train)
        test_accuracy = calculate_accuracy(test_outputs, y_test)
        # store the best model
        if test_accuracy > highest_accuracy:
            best_model = model
            highest_accuracy = test_accuracy

    if epoch % 10 == 0:
        print(
            f'''Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {train_accuracy:.4f} |
            Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}'''
        )

# save the model
torch.save(best_model.state_dict(), "models/doc2vec_model.pt")
