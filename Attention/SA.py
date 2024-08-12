import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        #初始化Q,K,V矩阵
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        attention_score = torch.matmul(q, k.transpose(-1,-2))/torch.sqrt(torch.tensor(float(self.dim)))
        attention_weight = F.softmax(attention_score, dim=-1)
        y = torch.matmul(attention_weight, v)
        return y
    
if __name__=='__main__':
    x = torch.randn(3,1024,64) # (B,N,C)
    model = SelfAttention(64)
    y = model(x)
    print(y.shape)
