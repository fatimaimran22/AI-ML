import torch   #PyTorch Library consists -------> tensors, neural networks, automatic differentiation, gradient descent, GPU Support
"""
loss.backward() ---> Backpropagation, chain rule, gradient.

Go backward through every calculation you remembered and compute all gradients.

chain: w -> y -> loss
now backward: loss -> y -> w

Input is Weight and Output is Loss, 
so Gradient --> if we increase w by tiny amount how much will loss increase by that tiny amount.

if we increase by tiny amount 0.001 the loss will increase by grad*0.001

"""

weight = torch.tensor([2.0], requires_grad = True) # #PyTorch will compute gradients. To learn gradient for weight
input = torch.tensor([3.0])

print("-" * 40)
print("Initial State:")
print(f"Weight: {weight.item()}")
print(f"Input: {input.item()}")


#------------------------------------------------------
# FORWARD PASS

output = weight * input
loss = output ** 2

print("-" * 40)
print("Forward Pass:")
print(f"Output = weight * input = {weight.item()} * {input.item()} = {output.item()}")
print(f"Loss   = output^2 = {output.item()}^2 = {loss.item()}")


#------------------------------------------------------
# Before Backward

print("-" * 40)
print("Before Backward:")
print(f"Gradient stored in weight.grad =  {weight.grad}")


#------------------------------------------------------
# Backward pass

print("-" * 40)
print("Backward pass:")

loss.backward()

print(f"Gradient stored in weight.grad =  {weight.grad.item()}")

# -----------------------------------
# See what happens with a tiny change
# -----------------------------------

tiny_change = 0.001

old_loss = loss.item()

new_weight = weight + tiny_change
new_output = input * new_weight
new_loss = new_output ** 2

actual_change = new_loss - old_loss
predicted_change = weight.grad.item() * tiny_change

print("\nTINY CHANGE EXPERIMENT")
print("-" * 40)
print(f"Increase weight by: {tiny_change}")
print(f"Old weight: {weight.item()}")
print(f"New weight: {new_weight.item()}")
print()
print(f"Old loss: {old_loss}")
print(f"New loss: {new_loss.item()}")
print()
print(f"Actual loss change    : {actual_change.item():.4f}")
print(f"Predicted by gradient : {predicted_change:.4f}")

# -----------------------------------
# Gradient Descent step
# -----------------------------------

learning_rate = 0.1

print("\nGRADIENT DESCENT STEP")
print("-" * 40)
print(f"Current weight : {weight.item()}")
print(f"Gradient       : {weight.grad.item()}")
print(f"Learning rate  : {learning_rate}")

with torch.no_grad():
    weight -= learning_rate * weight.grad

print(f"New weight     : {weight.item()}")

# -----------------------------------
# Compute new loss after update
# -----------------------------------

new_output = weight * input
new_loss = new_output ** 2

print("\nAFTER WEIGHT UPDATE")
print("-" * 40)
print(f"New output: {new_output.item()}")
print(f"New loss  : {new_loss.item()}")

# -----------------------------------
# Clear gradients (important in training loops)
# -----------------------------------

weight.grad.zero_()

print("\nAFTER CLEARING GRADIENTS")
print("-" * 40)
print(f"weight.grad = {weight.grad.item()}")

