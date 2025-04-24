# medical-Bioes
import csv
import random
from collections import defaultdict


def load_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]
    return data

def train_test_split(data, test_size=0.3):
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


def evaluate(predictions, truths):
    tp = fp = tn = fn = 0
    for p, t in zip(predictions, truths):
        if t == '1':
            if p == '1':
                tp += 1
            else:
                fn += 1
        else:
            if p == '1':
                fp += 1
            else:
                tn += 1
    accuracy = (tp + tn) / len(predictions)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return accuracy, fpr, fnr


def dummy_model_predict(data):
    return [random.choice(['0', '1']) for _ in data]


def simulate_csv_data():
    data = []
    for _ in range(100):
        data.append({
            'feature1': str(random.random()),
            'label': random.choice(['0', '1']),
            'gender': random.choice(['male', 'female']),
            'race': random.choice(['group1', 'group2'])
        })
    return data

data = simulate_csv_data()
train_data, test_data = train_test_split(data)
y_true = [row['label'] for row in test_data]
y_pred = dummy_model_predict(test_data)

overall_accuracy, _, _ = evaluate(y_pred, y_true)
print(f"Overall Accuracy: {overall_accuracy:.2f}")

for attr in ['gender', 'race']:
    print(f"\nFairness analysis by {attr}:")
    groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    for row, pred in zip(test_data, y_pred):
        group = row[attr]
        groups[group]['y_true'].append(row['label'])
        groups[group]['y_pred'].append(pred)
    for group, values in groups.items():
        acc, fpr, fnr = evaluate(values['y_pred'], values['y_true'])
        print(f"  {group}: Accuracy={acc:.2f}, FPR={fpr:.2f}, FNR={fnr:.2f}")

print("\nSuggested Mitigation Strategies:")
print("- Balance dataset manually based on demographics")
print("- Implement rule-based checks for group parity")
print("- Use controlled thresholding per group")

