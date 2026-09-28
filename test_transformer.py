import torch
from transformer import GPTConfig, Block

def test_block_shape():
    config = GPTConfig(n_embd=128, n_head=4, block_size=32, dropout=0.0)
    block = Block(config)
    
    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config.n_embd)
    
    y = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    assert y.shape == x.shape, "Output shape does not match input shape"
    print("Shape test passed!")

if __name__ == "__main__":
    test_block_shape()
