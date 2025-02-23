import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)

# Step 1: compute all attention scores
attn_scores = torch.empty(6,6)
attn_scores = inputs @ inputs.T # Replace the for loop in Python for inceasing performance
print(attn_scores)

# Step 2: compute all attention weights
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Verify the row indeeded sum to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# Step 3: compute all context vector
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)