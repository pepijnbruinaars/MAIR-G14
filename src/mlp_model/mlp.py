import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import pickle

# read the data
df = pd.read_csv("data/splits/train_dialog_acts_no_dupes.csv")
training_data = pd.read_csv("data/splits/rf_training_data.csv", index_col = 0)

df["label"] = pd.factorize(df["label"])[0]
df["label"] = df["label"].replace(-1,14)

x = training_data
y = df["label"]

x_train, x_test, y_train, y_test =  train_test_split(
        x, y, test_size=0.15, random_state=42
    )

x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)

y_train = torch.LongTensor(y_train.values)
y_test = torch.LongTensor(y_test.values)


# define classification model
class FeedForwardNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_rate):
        super(FeedForwardNN, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden_size)
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_3 = torch.nn.Linear(hidden_size, output_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.dropout(self.relu(self.layer_2(self.relu(self.layer_1(x))))))

# function to calculate accuracy
def calculate_accuracy(preds, targets):
    count = 0
    for pred, target in zip(preds, targets):
        if pred == target:
            count += 1

    return count / len(preds)

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# the final training loop
def training_loop(model, criterion, optimizer):
    epochs = 200
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
                f''' Accuracy: {train_accuracy:.4f}| Test Accuracy: {test_accuracy:.4f} | 
                epoch {epoch}'''
            )
    return best_model

# train the model with different hyperparameter values and see what works best          
VECTOR_SIZE = x_train.shape[1]
OUTPUT_SIZE = 15 # 15 different dialog acts
HIDDEN_SIZE = 75
DROPOUT_RATE = 0.2

model = FeedForwardNN(VECTOR_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, DROPOUT_RATE).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
best_model = training_loop(model,criterion,optimizer)



# save the model
torch.save(best_model.state_dict(), "models/mlp_model.pt")

def predict_single_input(input):
    
    df = pd.read_csv("data/no_duplicates_dialog_acts.csv")
    labels = pd.factorize(df["label"])[1]
    
    # load the vectorizer from data
    with open('models/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    # Load the model
    model = FeedForwardNN(VECTOR_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, DROPOUT_RATE)
    model.load_state_dict(torch.load("models/mlp_model.pt"))
    model.eval()

    # Add the input to the vectorized training data
    input_data = torch.FloatTensor(vectorizer.transform([input]).toarray())

    # Predict the intent of the input
    y_pred = torch.argmax(model(input_data))

    # Return the label of the intent
    return labels[y_pred.item()]

print(predict_single_input("Hello"))


