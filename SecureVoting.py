"""
Daemon Husk
2024-09-25

ASK NICELY
"""

import hashlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_p_adic_number(hash_value, p):
    # Generate a p-adic number using the hash value and p
    p_adic_number = np.array([int(hash_value[i:i+2], 16) for i in range(0, len(hash_value), 2)])
    return p_adic_number

def generate_fractal_space(p_adic_number, p):
    # Generate a fractal space using the p-adic number and p
    fractal_space = np.zeros((len(p_adic_number), len(p_adic_number)))
    for i in range(len(p_adic_number)):
        for j in range(len(p_adic_number)):
            fractal_space[i, j] = p_adic_number[i] * p_adic_number[j] % p
    return fractal_space

def generate_graph(fractal_space):
    # Generate a graph using the fractal space
    G = nx.Graph()
    for i in range(len(fractal_space)):
        for j in range(len(fractal_space)):
            if fractal_space[i, j] != 0:
                G.add_edge(i, j)
    return G

def encode_vote(voter_id, vote_choice, p):
    # Encode the vote using the voter's identifier and vote choice
    hash_value = hashlib.sha256((voter_id + vote_choice).encode()).hexdigest()
    p_adic_number = generate_p_adic_number(hash_value, p)
    fractal_space = generate_fractal_space(p_adic_number, p)
    G = generate_graph(fractal_space)
    return G

def verify_vote(voter_id, vote_choice, p):
    # Verify the vote using the voter's identifier and vote choice
    hash_value = hashlib.sha256((voter_id + vote_choice).encode()).hexdigest()
    p_adic_number = generate_p_adic_number(hash_value, p)
    fractal_space = generate_fractal_space(p_adic_number, p)
    G = generate_graph(fractal_space)
    return G

# Example usage
voter_id = "1234567890"
vote_choice = "A"
p = 23

encoded_vote = encode_vote(voter_id, vote_choice, p)
verified_vote = verify_vote(voter_id, vote_choice, p)

print("Encoded Vote:")
nx.draw(encoded_vote, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold')
plt.show()

print("Verified Vote:")
nx.draw(verified_vote, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold')
plt.show()
```
