"""
Text-to-Mathematical-Values Conversion Algorithm

This module provides a set of functions for converting text to various mathematical representations,
including Base31, Base3, binary, decimal, and hexadecimal. It also includes a graph generation
function that produces a power series generating function from a given graph.

Functions
---------
text_to_base31(text)
    Convert text to a Base31 string.

base31_to_base3(base31_text)
    Convert a Base31 string to a Base3 integer.

base3_to_binary(base3_value)
    Convert a Base3 integer to a binary string.

binary_to_decimal(binary_string)
    Convert a binary string to a decimal integer.

decimal_to_hex(decimal_value)
    Convert a decimal integer to a hexadecimal string.

calculate_prime_factors(decimal_value)
    Calculate the prime factors of a decimal integer.

calculate_trigonometric_values(decimal_value)
    Calculate the trigonometric values of a decimal integer.

GraphGenerator
-------------
A class for generating graphs and producing power series generating functions.

Methods
-------
generate_power_series()
    Generate a power series generating function from the graph.

visualize_graph()
    Visualize the graph using NetworkX and Matplotlib.

test_algorithm(text_corpus)
    Test the text-to-mathematical-values conversion algorithm on a given text corpus.

Variables
---------
base31_chars
    The Base31 character set.

Author
------
Daemon Husk (Dave)

"""

import string
import math
import networkx as nx
import matplotlib.pyplot as plt
from sympy import symbols, prod
import time

# Define the Base31 character set
base31_chars = string.digits + string.ascii_uppercase + '!@#$%^&*()'

def text_to_base31(text):
    # Create a dictionary to map words/symbols to Base31 characters
    words = text.split()
    unique_words = list(set(words))
    char_map = {word: base31_chars[i] for i, word in enumerate(unique_words)}
    # Convert the text to Base31 string
    base31_text = ''.join(char_map[word] for word in words)
    return base31_text, char_map

def base31_to_base3(base31_text):
    # Convert Base31 string to Base3 integer
    base3_value = 0
    for char in base31_text:
        base3_value = base3_value * 31 + base31_chars.index(char)
    return base3_value

def base3_to_binary(base3_value):
    # Convert Base3 integer to binary string
    binary_string = ''
    while base3_value > 0:
        binary_string = str(base3_value % 2) + binary_string
        base3_value //= 2
    # Pad the binary string with leading zeros if necessary
    while len(binary_string) % 3 != 0:
        binary_string = '0' + binary_string
    return binary_string

def binary_to_decimal(binary_string):
    # Convert binary string to decimal integer
    decimal_value = int(binary_string, 2)
    return decimal_value

def decimal_to_hex(decimal_value):
    # Convert decimal integer to hexadecimal string
    hex_string = hex(decimal_value)[2:].upper()
    return hex_string

def calculate_prime_factors(decimal_value):
    # Calculate prime factors of a decimal integer
    factors = []
    divisor = 2
    while divisor * divisor <= decimal_value:
        if decimal_value % divisor == 0:
            factors.append(divisor)
            decimal_value //= divisor
        else:
            divisor += 1
    if decimal_value > 1:
        factors.append(decimal_value)
    return factors

def calculate_trigonometric_values(decimal_value):
    # Calculate trigonometric values of a decimal integer (in radians)
    angle_rad = math.radians(decimal_value)
    sin_value = math.sin(angle_rad)
    cos_value = math.cos(angle_rad)
    tan_value = math.tan(angle_rad)
    return sin_value, cos_value, tan_value

class GraphGenerator:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(vertices)
        self.graph.add_edges_from(edges)
        self.variables = symbols(f'x:{len(vertices)}')

    def generate_power_series(self):
        terms = []
        for edge in self.edges:
            i, j = self.vertices.index(edge[0]), self.vertices.index(edge[1])
            terms.append(1 + self.variables[i] * self.variables[j])
        power_series = prod(terms)
        return power_series

    def visualize_graph(self):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold')
        labels = {edge: f'({self.variables[self.vertices.index(edge[0])]}, {self.variables[self.vertices.index(edge[1])]})' for edge in self.edges}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_size=10)
        plt.axis('off')
        plt.show()

def test_algorithm(text_corpus):
    # Split the corpus into training and test sets
    train_size = int(0.8 * len(text_corpus))
    train_corpus = text_corpus[:train_size]
    test_corpus = text_corpus[train_size:]

    # Test the text-to-mathematical-values conversion
    train_time = 0
    test_time = 0
    train_size_ratio = 0
    test_size_ratio = 0

    for text in train_corpus:
        start_time = time.time()
        base31_text, _ = text_to_base31(text)
        base3_value = base31_to_base3(base31_text)
        binary_string = base3_to_binary(base3_value)
        decimal_value = binary_to_decimal(binary_string)
        hex_string = decimal_to_hex(decimal_value)
        train_time += time.time() - start_time
        train_size_ratio += len(hex_string) / len(text)

    for text in test_corpus:
        start_time = time.time()
        base31_text, _ = text_to_base31(text)
        base3_value = base31_to_base3(base31_text)
        binary_string = base3_to_binary(base3_value)
        decimal_value = binary_to_decimal(binary_string)
        hex_string = decimal_to_hex(decimal_value)
        test_time += time.time() - start_time
        test_size_ratio += len(hex_string) / len(text)

    train_time /= len(train_corpus)
    test_time /= len(test_corpus)
    train_size_ratio /= len(train_corpus)
    test_size_ratio /= len(test_corpus)

    print(f"Average training time: {train_time:.6f} seconds")
    print(f"Average test time: {test_time:.6f} seconds")
    print(f"Average training size ratio: {train_size_ratio:.2f}")
    print(f"Average test size ratio: {test_size_ratio:.2f}")

    # Test the graph generation
    vertices = ['v1', 'v2', 'v3', 'v4']
    edges = [('v1', 'v2'), ('v2', 'v3'), ('v3', 'v4'), ('v4', 'v1')]
    generator = GraphGenerator(vertices, edges)
    power_series = generator.generate_power_series()
    print("Power Series Generating Function:")
    print(power_series)
    generator.visualize_graph()

# Example usage
text_corpus = ["The Riemann Zeta Function (ζ(s)) is a 3D surface",
               "The quick brown fox jumps over the lazy dog",
               "Data compression is an important topic in computer science",
               "Graphs and networks are widely used in various fields"]
test_algorithm(text_corpus)

results = """
Average training time: 0.000000 seconds
Average test time: 0.000000 seconds
Average training size ratio: 0.21
Average test size ratio: 0.21
Power Series Generating Function:
(x0*x1 + 1)*(x0*x3 + 1)*(x1*x2 + 1)*(x2*x3 + 1)
"""
