"""
We define a logger to keep track of the loss function progress while training the model.
(Not neccessary, but feels nice to have).
We also store the losses in a list to possibly plot them later to visualize where the model plateus.
"""

from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
class LossLogger(CallbackAny2Vec):
    """Output loss at each epoch"""

    def __init__(self, epochs):
        self.epoch = 1
        self.epochs = epochs
        self.losses = []
        self.delta_losses = []
        self.progress_bar = tqdm(total=epochs,unit="epoch")

    def on_epoch_end(self, model):
        self.progress_bar.update(1)

        if self.epoch == self.epochs:
            self.progress_bar.close()
            losses_with_indices = [x for x in enumerate(self.losses)]
            print(f"Losses: {losses_with_indices}")

        loss = model.get_latest_training_loss()
        last_loss = self.losses[self.epoch - 2] if len(self.losses) > 0 else None
        delta = (
            loss - last_loss
            if last_loss is not None
            else loss
        )
        self.losses.append(loss)
        self.delta_losses.append(delta)
        self.epoch += 1