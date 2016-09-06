
"""
# 6.034 Lab 6 2015: Neural Nets & SVMs
# Designed and implemented by Rafal Jankowski, Nov 2015

Below is implementation of forward propagation and backpropagation using chain rule on a custom-built Neural Network API
It starts with a number of helper functions for calculating accuracy, updating weights/deltas and then
it pulls them into back_prop and back_online for stacked and online backpropagation algorithm.

Example of implementation:
print nn_basic #imported from nn_problems.py
inputs = {'in1': 10, 'in3': 20} # set sample inputs
print forward_prop(nn_basic, inputs) # sample of output using forward propagation (with selected threshold function)
print back_prop(nn_basic, inputs, 5) # solve using inputs for output of 5
"""

from nn_problems import *
from math import e


# Wiring a neural net
nn_half = [1]
nn_angle = []
nn_cross = []
nn_stripe = []
nn_hexagon = []
TEST_NN_GRID = False
nn_grid = []

#  ------------------------ Helper functions --------------------------------

def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    if x >= threshold:
        return 1
    else:
        return 0
        
def sigmoid(x, steepness=1, midpoint=0):
    sigmoid = 1/(1 + e**(-steepness * (x - midpoint)))
    return float(sigmoid)

def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5*(desired_output-actual_output)**2
    
# Forward propagation with a specific threshold function on the final node (stairstep or sigmoid)
def forward_prop(net, input_values, threshold_fn=stairstep):
    inputs = input_values
    inputs_new = {}
    nodes = net.topological_sort()
    for node in nodes:
        output = 0.00
        wires = net.get_incoming_wires(node)
        for wire in wires:
            if type(wire.startNode) == str:
                try:
                    output += inputs[wire.startNode] * wire.weight
                except:
                    output += inputs_new[wire.startNode] * wire.weight
            else:
                output += wire.startNode * wire.weight
                    
        output = threshold_fn(output)
        inputs_new[node] = output
        if node == nodes[-1]:
            return (output, inputs_new)

# Backward propagation
def calculate_deltas(net, input_values, desired_output):
    deltas = {}
    nodes = net.topological_sort()
    (output, outputs) = forward_prop(net, input_values, sigmoid)
    
    
    for node in reversed(nodes):
        if net.is_output_neuron(node):
            delta_B = output*(1-output)*(desired_output - output)
            deltas[node] = delta_B
        else:
            out_nodes = net.get_outgoing_neighbors(node)
            out_wires = net.get_outgoing_wires(node)
            node_sum = 0
            for wire in out_wires:
                weight = wire.weight
                out_node = wire.endNode
                out_output = outputs[out_node]
                out_delta = deltas[out_node]
                node_sum += weight*out_delta # *out_output
            delta_B = outputs[node]*(1-outputs[node])*node_sum
            deltas[node] = delta_B
    return deltas
    
#Update weights on the net using the deltas calculated above
def update_weights(net, input_values, desired_output, r=1):
    net2 = NeuralNet(net.inputs, net.neurons)
    deltas = calculate_deltas(net, input_values, desired_output)
    output, outputs = forward_prop(net, input_values, sigmoid)
    nodes = net.topological_sort()
    for node in reversed(nodes):
        out_wires = net.get_incoming_wires(node)
        for wire in out_wires:
            if type(wire.startNode) == str:
                if wire.startNode in input_values:
                    weight = wire.weight + r * deltas[wire.endNode] * input_values[wire.startNode]
                else:
                    weight = wire.weight + r * deltas[wire.endNode] * outputs[wire.startNode]
            else:
                weight = wire.weight + r * deltas[wire.endNode] * wire.startNode
            net2.join(wire.startNode, wire.endNode, weight)
    net2.join(net.get_output_neuron(), NeuralNet.OUT)
    return net2
# ----------------------------------------------------


# ------------------ Implement backpropagation optimization using the helper functions above -----------------
def back_prop(net, input_values, desired_output, r=1, accuracy_threshold=-.0):
    i = 1
    net = update_weights(net, input_values, desired_output, r)
    (actual_output, neuron_values) = forward_prop(net, input_values, sigmoid)
    while (accuracy(desired_output, actual_output)) < accuracy_threshold and i <50:
        net = update_weights(net, input_values, desired_output, r)
        (actual_output, neuron_values) = forward_prop(net, input_values)
        i += 1
    return (net, i)
    
parameters = [ [{'x':0, 'y':0},0], [{'x':0, 'y':1},1], [{'x':1, 'y':0},1]]

def back_online(net, parameters, r=1, accuracy_threshold=-.0):
    i = 1
    for set in parameters:
        input_values = set[0]
        desired_output = set[1]
        net = update_weights(net, input_values, desired_output, r)
        (actual_output, neuron_values) = forward_prop(net, input_values, sigmoid)
        while (accuracy(desired_output, actual_output)) < accuracy_threshold and i <50:
            net = update_weights(net, input_values, desired_output, r)
            (actual_output, neuron_values) = forward_prop(net, input_values)
            i += 1
    return (net, i)

# ---------------------------------------------------------------------------------------------



# ************************************ Example implementation *****************************************************
print nn_basic #imported from nn_problems.py
inputs = {'in1': 10, 'in3': 20} # set sample inputs
print forward_prop(nn_basic, inputs) # sample of output using forward propagation (with selected threshold function)
print back_prop(nn_basic, inputs, 5) # solve using inputs for output of 5




NAME = 'Rafal Jankowski'
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = 18
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
