import torch
from attention.head import AttentionHead, MultiHeadAttention

att = MultiHeadAttention(num_heads=8, input_dim=32 * 32, query_dim=512, head_dim=512)

print(f"Total number of parameters: {att.total_parameters}")

input_seq = torch.randn((10, 32 * 32))
labels = torch.randn((10, 1))
with torch.no_grad():
    frozen_output_seq = att(input_seq)
output_seq = att(input_seq)
decoder = torch.nn.Linear(32 * 32, 1)
assert(output_seq.shape == torch.Size((10, 32 * 32)))

criterion = torch.nn.MSELoss(reduction='mean')
preds = decoder(output_seq)
loss = criterion(labels, preds)
loss.backward()

print(f"Input is {input_seq}")
print(f"Output is {frozen_output_seq}")
print(f"Total loss for {10} tokens is {10 * loss.item()}")
for i, head in enumerate(att.heads):
    print(f"Head {i + 1}:")
    for layer, name in zip((head.query, head.key, head.value), ("query", "key", "value")):
        print(f"    Grad for {name} matrix has L2 norm {torch.linalg.norm(layer.weight.grad, ord=2)}")
