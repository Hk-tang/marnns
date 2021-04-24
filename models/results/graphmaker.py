import matplotlib.pyplot as plt

losses = [0.0021003958930218857, 0.000621866107647149, 0.0003479763391509221, 0.00023035754907106313, 0.00018852806592182974]

plt.plot(losses)
plt.title("Losses per epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()