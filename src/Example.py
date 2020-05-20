from src.InventorySimulator import InventorySystem
import torch
from src.DeepQ_InventorySystem import deepQ_inventorysystem
from matplotlib import pylab as plt

# Set device to CPU
device = torch.device("cpu")

mean_daily_demand = 20
sd_demand = 1
daily_invcost = 0.01  # per unit per day
daily_stockoutcost = 4  # per unit per day
reorder_costs = 2
system = InventorySystem(mean_daily_demand, sd_demand, daily_invcost, daily_stockoutcost, reorder_costs)

costs, losses, order_level = deepQ_inventorysystem(system=system, epochs=500, time_per_epoch=40)
plt.plot(order_level)
plt.show()
print(order_level)
