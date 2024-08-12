import torch
import torch.nn as nn

class MutiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, group_num) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        self.group_num = group_num

        # 初始化Q,K,V矩阵
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, self.group_num*self.head_dim)
        self.v_linear = nn.Linear(dim, self.group_num*self.head_dim)

        # 输出线性层
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, x, attention_mask=None):
        batch_size = x.shape[0]

        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.split_head(q)
        k = self.split_head(k, self.group_num)
        v = self.split_head(v, self.group_num)

        attention_score = torch.matmul(q,k.transpose(-1,-2))/torch.sqrt(torch.tensor(float(self.head_dim)))
        
        if attention_mask != None:
            attention_score += attention_mask * -1e-9

        attention_wight = torch.softmax(attention_score, dim=-1)
        y = torch.matmul(attention_wight, v)
        # 注意力输出拼接
        y = y.transpose(1,2).contiguous().view(batch_size, -1, self.head_dim*self.num_heads)
        y = self.out_linear(y)
        return y

    def split_head(self, x, group_num=None):
        batch_size, N = x.shape[:2]
        if group_num==None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        else:  
            x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1,2)
            x = x[:,:,None,:,:].expand(batch_size, group_num, self.num_heads//group_num, N, self.head_dim).reshape(batch_size, self.num_heads, N, self.head_dim)
            return x
        

if __name__=='__main__':
    x = torch.randn(3,1024,64) # (B,N,C)
    model = MutiHeadAttention(64, 4, 2)
    y = model(x)
    print(y.shape)