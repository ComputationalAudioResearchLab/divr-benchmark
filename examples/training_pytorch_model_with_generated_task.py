import torch
from torch import nn
from divr_diagnosis import diagnosis_maps
from divr_benchmark import Benchmark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diag_map = diagnosis_maps.CaRLab_2025()
benchmark = Benchmark(
    storage_path="/home/user/storage",
    version="v1",
    sample_rate=16000,
)
task = benchmark.load_task(
    task_path="/home/user/svd_a_to_meei_test",
    diag_level=None,
    diagnosis_map=diag_map,
    load_audios=True,
)

model = nn.RNN(
    input_size=1,
    hidden_size=32,
    num_layers=2,
).to(device)

def to_long_tensor(data):
    return torch.tensor(data, dtype=torch.long, device=device)

def to_float_tensor(data):
    return torch.tensor(data, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(
    weight=to_float_tensor(task.train_class_weights())
)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-5)
num_epochs = 100

for epoch in range(num_epochs):
    train_loss = 0
    train_points = 0
    val_loss = 0
    val_points = 0
    # Training
    for train_point in task.train:
        actual_label = to_long_tensor(task.diag_to_index(train_point.label, level=None))
        for audio in train_point.audio:
            audio = to_float_tensor(audio)
            _, predicted_label = model(audio)
            predicted_label = predicted_label[-1]
            loss = criterion(predicted_label, actual_label)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            train_points += 1
    # Validation
    with torch.no_grad():
        for val_point in task.val:
            actual_label = to_long_tensor(task.diag_to_index(val_point.label, level=None))
            for audio in val_point.audio:
                audio = to_float_tensor(audio)
                _, predicted_label = model(audio)
                predicted_label = predicted_label[-1]
                loss = criterion(predicted_label, actual_label)
                val_loss += loss.item()
                val_points += 1
    print(f"Loss:: train={train_loss/train_points}, val={val_loss/val_points}")


test_correct = 0
test_total = 0
with torch.no_grad():
    for test_point in task.test:
        actual_label = to_long_tensor(task.diag_to_index(test_point.label, level=None))
        for audio in test_point.audio:
            audio = to_float_tensor(audio)
            _, predicted_label = model(audio)
            predicted_label = predicted_label[-1]
            test_total += 1
            test_correct += int(predicted_label == actual_label)
print(f"Total test accuracy = {test_correct/test_total}")
