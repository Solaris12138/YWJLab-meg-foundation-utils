import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NTXentPolyLoss(nn.Module):

    def __init__(self, batch_size, temperature, device, epsilon=1.0):
        """
        Normalized temperature-scaled cross entropy loss

        Parameters
        ----------
        batch_size : int
            The size of mini batch.
        temperature : float
            Scaling factor
        epsilon : float
            The coefficient of the first-order polynomial.

        """
        super(NTXentPolyLoss, self).__init__()
        
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.epsilon = epsilon

        self.mask = self._init_mask().to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.similarity_func = nn.CosineSimilarity(dim=-1)
        self.labels = torch.zeros(2 * self.batch_size).long().to(self.device)

        self.to(self.device)
    
    def _init_mask(self):
        mask = torch.from_numpy(
            (
                np.eye(2 * self.batch_size) +\
                np.eye(2 * self.batch_size, k=self.batch_size) +\
                np.eye(2 * self.batch_size, k=-self.batch_size)
            )
        )
        mask = (1 - mask).type(torch.bool)
        
        return mask
    
    def _cosine_similarity(self, x, y):
        return self.similarity_func(x.unsqueeze(1), y.unsqueeze(0))
    
    def forward(self, embedding, embedding_aug):
        """
        Forward function

        Parameters
        ----------
        embedding : torch.Tensor, shape (batch_size, embedding_dim)
            The embedding of original sample.
        embedding_aug : torch.Tensor, shape (batch_size, embedding_dim)
            The embedding of augmentation sample.

        Returns
        -------
        loss : torch.Tensor, shape (1, )
            The NTXent loss with poly loss correction.

        """
        embedding_ = torch.cat((embedding, embedding_aug), dim=0)
        similarity_matrix = self._cosine_similarity(embedding_, embedding_)

        positives = torch.cat(
            (
                torch.diag(similarity_matrix, self.batch_size),
                torch.diag(similarity_matrix, -self.batch_size),
            )
        ).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        
        ce = self.criterion(logits, self.labels)
        pt = torch.mean(F.softmax(logits, dim=-1)[:, 0])
        
        loss = ce + self.epsilon * (1.0 - pt)

        return loss