from src.InventorySimulator import InventorySystem
import numpy as np
import torch
from torch.autograd import Variable
import random
from matplotlib import pylab as plt


def deepQ_inventorysystem(system, epochs, time_per_epoch):
    inputl = 1
    hiddenl1 = 2
    hiddenl2 = 2
    outputl = 2

    model = torch.nn.Sequential(
        torch.nn.Linear(inputl, hiddenl1),  # input layer to hidden layer 1
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hiddenl1, hiddenl2),  # hidden layer 1 to hidden layer 2
        torch.nn.LeakyReLU(),
        torch.nn.Linear(hiddenl2, outputl))  # hidden layer 2 to output layer

    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gamma = 0.95
    epsilon = 0.4

    # Training
    losses = []  # A
    batchSize = 4
    buffer = 10
    replay = []
    costlist = []
    order_level = []
    action_set = {
        0: 'not_order',
        1: 'order'}

    for i in range(epochs):  # B
        state_ = system.init_state()  # D
        state = Variable(torch.from_numpy(state_).float())  # E
        # state = state.to(device)
        status = 1  # F
        totalcost = 0
        mov = 0
        while status == 1:  # G
            mov += 1
            # state = state.to(device)
            qval = model(state)  # H
            qval_ = qval.data.numpy()

            if random.random() < epsilon:  # I
                action_ = np.random.randint(0, 2)
            else:
                action_ = (np.argmin(qval_))


            action = action_set[action_]  # J



            new_state_ = system.new_state(action)  # K
            new_state = Variable(torch.from_numpy(new_state_).float())  # L
            cost = system.costs()
            totalcost += cost
            print(qval_, action, cost, state, new_state_)

            if action_ == 1:
                order_level.append(state)

            if len(replay) < buffer:
                replay.append((state, action_, cost, new_state))
            else:
                replay.pop(0)
                replay.append((state, action_, cost, new_state))
                minibatch = random.sample(replay, batchSize)
                output = Variable(torch.empty(batchSize, 2, dtype=torch.float))
                target = Variable(torch.empty(batchSize, 2, dtype=torch.float))
                h = 0
                for memory in minibatch:
                    old_state, action_m, cost_m, new_state_m = memory
                    old_qval = model(old_state)
                    # new_state_m = new_state_m.to(device)
                    newQ = model(new_state_m).data.numpy()
                    minQ = np.min(newQ)
                    y = torch.zeros((1, 2))
                    y[:] = old_qval[:]

                    update = (cost_m + (gamma * minQ))

                    y[0][action_m] = update
                    output[h] = old_qval
                    target[h] = Variable(y)
                    h += 1

                # X_train = X_train.to(device)
                # y_train = X_train.to(device)
                loss = loss_fn(output, target)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.data)
                optimizer.step()

            state = new_state

            if mov > time_per_epoch:
                status = 0
                mov = 0
                costlist.append(totalcost)
                print('epoch ', i)

            if epsilon > 0.001:
                epsilon -= (1 / epochs)

    return costlist, losses, order_level
