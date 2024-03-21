import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0
    for (model_input, sam_features) in train_loader:
        model_input = model_input.cuda()
        sam_features = sam_features.cuda()

        output = model(model_input)

        loss = loss_fn(output, sam_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        del model_input
        del sam_features
        
    return model, running_loss


def train_model(model, train_loader, test_loader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    model = model.cuda()
    best_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        model.train()
        model, epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn)

        if epoch_loss < best_loss:
            save_checkpoint(model, optimizer, 'compress_block', epoch)

    return dict(train_losses), dict(test_losses)


def save_checkpoint(model, optimizer, filename, EPOCH):
    name_checkpoint = f'{name}.pth'
    save_path = os.path.join(os.getcwd(), name_checkpoint)
    with open(filename, "wb") as fp:
        torch.save(model.state_dict(), fp)

        torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)