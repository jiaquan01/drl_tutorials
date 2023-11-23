import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


#Hyperparameters
learning_rate = 5.0e-3 
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=0)
    
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        dis_return = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            dis_return = r + gamma * dis_return
            loss = -torch.log(prob) * dis_return
            loss.backward()
        self.optimizer.step()
        self.data = []
    

def main():
    env = gym.make('CartPole-v1') #, render_mode="human"
    torch.manual_seed(0)
    pi = Policy()
    total_reward = 0.0
    print_interval = 20
    
    for n_epi in range(1000): 
        state, _ = env.reset(seed=123)
        terminated,truncated = False,False
        while not (terminated | truncated)  : # CartPole-v1 forced to terminates at 500 step.           
            prob = pi(torch.from_numpy(state).float())
            #print(prob)
            pd = Categorical(prob) 
            action = pd.sample() 
            #print(action)
            state_prime, reward, terminated, truncated, info = env.step(action.item())

            pi.put_data((reward,prob[action])) 
            state = state_prime
            total_reward += reward
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, total_reward : {}".format(n_epi, total_reward/print_interval))
            total_reward = 0.0

    env.close()

    
    
if __name__ == '__main__':
    main()