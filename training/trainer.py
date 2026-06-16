
'''
    Class for training the FDI localization model. 


'''

import torch


class Trainer:

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader):
        '''  Train the model for one epoch   '''

        self.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def evaluate_epoch(self, val_loader):
        '''  Evaluate the model on the validation set for one epoch   '''

        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs):
        '''  Fit the model for a given number of epochs   '''

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            val_loss = self.evaluate_epoch(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(
                f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return train_losses, val_losses

    def save_model(self, path):
        '''  Save the trained model to a file   '''
        torch.save(self.model.state_dict(), path)

        # if model is saved, print a message
        if torch.save(self.model.state_dict(), path):
            print(f'Model saved to {path}')
