import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import KMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS_BASELINE = 20
EPOCHS_EXTENDED = 50
PATIENCE = 3

# Transforms & DataLoaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_ds = KMNIST(root='./data', train=True,  transform=transform, download=True)
test_ds  = KMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# --------------------------------------------------------------------------------
# Model Definitions
# --------------------------------------------------------------------------------
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MLP40(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 40),
            nn.ReLU(),
            nn.Linear(40, 10)
        )
    def forward(self, x):
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class CNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.features(x)
        return self.classifier(x)

class CNN_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2), nn.ReLU(),
            nn.Conv2d(16,32,5,padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64,10)
    def forward(self,x):
        x = self.features(x).view(x.size(0), -1)
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.ReLU(),
            nn.Conv2d(out_ch,out_ch,3,padding=1)
        )
        self.skip = nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
    def forward(self,x):
        return nn.ReLU()(self.conv(x) + self.skip(x))

class CNN_C(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualBlock(1,16)
        self.pool1  = nn.MaxPool2d(2)
        self.layer2 = ResidualBlock(16,32)
        self.pool2  = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*7*7,64), nn.ReLU(), nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        return self.classifier(x)

# --------------------------------------------------------------------------------
# Training / Evaluation Functions
# --------------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_correct = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        preds = model(X)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss   += loss.item() * X.size(0)
        total_correct+= (preds.argmax(1)==y).sum().item()
    n = len(loader.dataset)
    return total_loss/n, total_correct/n

def eval_model(model, loader, criterion):
    model.eval()
    total_loss = total_correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X)
            total_loss   += criterion(preds, y).item() * X.size(0)
            total_correct+= (preds.argmax(1)==y).sum().item()
    n = len(loader.dataset)
    return total_loss/n, total_correct/n

def train_with_early_stopping(model, loaders, criterion, optimizer, epochs, patience):
    best_val = float('inf')
    counter = 0
    history = {'train_loss':[], 'val_loss':[], 'val_acc':[]}

    print(f"Training {model.__class__.__name__} with early stopping...")
    for ep in range(1, epochs+1):
        # train
        train_loss = 0
        model.train()
        for X, y in loaders['train']:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(loaders['train'].dataset)

        # validate
        val_loss = correct = 0
        model.eval()
        with torch.no_grad():
            for X, y in loaders['val']:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_loss  += criterion(out, y).item() * X.size(0)
                correct   += (out.argmax(1)==y).sum().item()
        val_loss /= len(loaders['val'].dataset)
        val_acc = correct / len(loaders['val'].dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best.pt')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load('best.pt'))
    return history

# --------------------------------------------------------------------------------
# Baseline Training (Linear, MLP40, SimpleCNN)
# --------------------------------------------------------------------------------
baseline_models = {
    'Linear': LinearModel().to(DEVICE),
    'MLP40': MLP40().to(DEVICE),
    'SimpleCNN': SimpleCNN().to(DEVICE)
}
histories_baseline = {}
for name, model in baseline_models.items():
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    h = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}
    print(f"\nTraining {name} model...")
    for epoch in range(1, EPOCHS_BASELINE+1):
        tl, ta = train_epoch(model, train_loader, crit, opt)
        vl, va = eval_model(model, test_loader, crit)
        h['train_loss'].append(tl)
        h['train_acc'].append(ta)
        h['test_loss'].append(vl)
        h['test_acc'].append(va)
        print(f"Epoch {epoch}: TrainLoss={tl:.3f}, TrainAcc={ta:.3f}, ValLoss={vl:.3f}, ValAcc={va:.3f}")
    histories_baseline[name] = h

# --------------------------------------------------------------------------------
# Extended CNN Variants with Early Stopping
# --------------------------------------------------------------------------------
fixed_lr = 5e-4
fixed_bs = 128

histories_extended = {}
best_overall = {'model':None, 'hist':None, 'acc':0, 'params':None}

for model_cls in [CNN_A, CNN_B, CNN_C]:
    tr_loader = DataLoader(train_ds, batch_size=fixed_bs, shuffle=True)
    va_loader = DataLoader(test_ds,  batch_size=fixed_bs)
    mdl = model_cls().to(DEVICE)
    opt = optim.Adam(mdl.parameters(), lr=fixed_lr)
    crit = nn.CrossEntropyLoss()

    hist = train_with_early_stopping(
        mdl,
        {'train': tr_loader, 'val': va_loader},
        crit, opt,
        epochs=EPOCHS_EXTENDED,
        patience=PATIENCE
    )
    histories_extended[model_cls.__name__] = hist

    _, final_acc = eval_model(mdl, test_loader, crit)
    print(f"{model_cls.__name__}: final test acc = {final_acc:.4f}")
    if final_acc > best_overall['acc']:
        best_overall.update({
            'model': model_cls.__name__,
            'hist': hist,
            'acc': final_acc,
            'params': (fixed_lr, fixed_bs)
        })

print("\nFinished training all extended CNNs.")
print(f"Best model: {best_overall['model']} with test accuracy {best_overall['acc']:.4f} and params {best_overall['params']}")

# --------------------------------------------------------------------------------
# Plot Baseline Results
# --------------------------------------------------------------------------------
for name, h in histories_baseline.items():
    epochs = range(1, EPOCHS_BASELINE+1)
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, h['train_loss'], label='Train Loss')
    plt.plot(epochs, h['test_loss'],  label='Test Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, h['train_acc'], label='Train Acc')
    plt.plot(epochs, h['test_acc'],  label='Test Acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------
# Plot Extended CNN Results
# --------------------------------------------------------------------------------
for name, h in histories_extended.items():
    epochs_ext = range(1, len(h['train_loss'])+1)
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs_ext, h['train_loss'], label='Train Loss')
    plt.plot(epochs_ext, h['val_loss'],   label='Val Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_ext, h['val_acc'],    label='Val Acc')
    plt.title(f'{name} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()
