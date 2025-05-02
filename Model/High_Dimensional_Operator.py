import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import math
# detect available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load data from a npz file
def load_data_from_npz(file_path):
    with np.load(file_path) as data:
        u_x1 = data['u_x1']  
        temp = data['temp']  
        y1 = data['y1']    
        u_x2_x3 = data['u_x2_x3']
        y2_y3 = data['features']
        G_u = data['G_u']  
    
    branch_inputs_x1_0 = torch.tensor(u_x1, dtype=torch.float32)
    branch_inputs_x1_temp = torch.tensor(temp/1000,dtype=torch.float32)
    branch_inputs_x1 = torch.stack([branch_inputs_x1_0,branch_inputs_x1_temp],dim=-2)
    trunk_inputs_y1 = torch.tensor(y1, dtype=torch.float32)
    trunk_inputs_y1 = trunk_inputs_y1.unsqueeze(1)
    #print(f"shapes,1D ,{branch_inputs_x1.shape},{trunk_inputs_y1.shape}")

    branch_inputs_x2_x3 = torch.tensor(u_x2_x3, dtype=torch.float32)
    trunk_inputs_y2_y3 = torch.tensor(y2_y3, dtype=torch.float32)
    #print(f"shapes,2D ,{branch_inputs_x2_x3.shape},{trunk_inputs_y2_y3.shape}")
    
    outputs = torch.tensor(G_u, dtype=torch.float32)
    outputs = outputs.unsqueeze(1)
    #print(f"shapes,out ,{outputs.shape}")

    return branch_inputs_x1, trunk_inputs_y1, branch_inputs_x2_x3, trunk_inputs_y2_y3, outputs

# write the log into a log_file
def write_log(log_file, message, i):
    mode = 'w' if i==0 else 'a'
    with open(log_file, mode) as f:
        f.write(message + '\n')

############## MACHINE LEARNING MODEL ##############
# Trunk Net
class Trunk_Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size_rho, activation=nn.LeakyReLU):
        super(Trunk_Net, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation())
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation())
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size_rho))
        layers.append(activation())  # Optional: remove this if no activation needed on the output
        
        # Combine layers into a Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Branch Net
class Branch_Net_Con_1D(nn.Module):
    def __init__(self, input_channels, conv_configs, output_size_rho, activation, kernel_size, pool_size, dropout_rate=0.5, n_no_pool=2):
        super(Branch_Net_Con_1D, self).__init__()

        layers = []
        in_channels = input_channels
        input_size = 600

        for i, out_channels in enumerate(conv_configs):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same", padding_mode='reflect'))
            layers.append(activation())

            if i >= n_no_pool:
                layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_size))
            
            layers.append(nn.Dropout(dropout_rate))
            
            in_channels = out_channels 

        self.conv_layers = nn.Sequential(*layers)

        conv_output_size = input_size
        for i in range(len(conv_configs)):
            if i >= n_no_pool: 
                conv_output_size = math.floor(conv_output_size / pool_size)

        conv_output_shape = (conv_configs[-1], conv_output_size)
        #print(f"conv_output_shape: {conv_output_shape}")

        # Fully connected
        self.fc1 = nn.Linear(torch.prod(torch.tensor(conv_output_shape)), output_size_rho)
        self.dropout = nn.Dropout(dropout_rate) 
        self.fc2 = nn.Linear(output_size_rho, output_size_rho)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Branch_Net_Con_2D(nn.Module):
    def __init__(self, input_channels, conv_configs, output_size_ori, activation, kernel_size, pool_size, dropout_rate=0.5, n_no_pool=2):
        super(Branch_Net_Con_2D, self).__init__()

        layers = []
        in_channels = input_channels
        input_2D_size = (30, 30)

        for i, out_channels in enumerate(conv_configs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same", padding_mode="circular"))
            layers.append(activation())

            if i >= n_no_pool:
                layers.append(nn.AvgPool2d(kernel_size=pool_size, stride=pool_size))

            layers.append(nn.Dropout(dropout_rate)) 
            
            in_channels = out_channels 

        self.conv_layers = nn.Sequential(*layers)

        conv_output_H, conv_output_W = input_2D_size
        for i in range(len(conv_configs)):
            if i >= n_no_pool: 
                conv_output_H = math.floor(conv_output_H / pool_size)
                conv_output_W = math.floor(conv_output_W / pool_size)

        conv_output_shape = (conv_configs[-1], conv_output_H, conv_output_W)
        #print(f"conv_output_shape: {conv_output_shape}")

        self.fc1 = nn.Linear(torch.prod(torch.tensor(conv_output_shape)), output_size_ori)
        self.dropout = nn.Dropout(dropout_rate) 
        self.fc2 = nn.Linear(output_size_ori, output_size_ori)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Deep Operator Network
class DeepONet(nn.Module):
    def __init__(self, branch_net, trunk_net, num_outputs=1):
        super(DeepONet, self).__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.num_outputs = num_outputs
        self.b = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        )

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,bi->b", x_func, x_loc)
        y = torch.unsqueeze(y, dim=1)
        y += self.b[index]
        return y

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        branch_output = self.branch_net(x_func)
        trunk_output = self.trunk_net(x_loc)
        y = self.merge_branch_trunk(branch_output, trunk_output, 0)
        return y

class CombineNet(nn.Module):
    def __init__(self, DeepONet1, DeepONet2):
        super(CombineNet, self).__init__()
        self.DeepONet1 = DeepONet1
        self.DeepONet2 = DeepONet2

    def forward(self, inputs1, inputs2):
        output1 = self.DeepONet1(inputs1)
        output2 = self.DeepONet2(inputs2)
        
        combined_output = output1 * output2
        
        return combined_output

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Net parameter setting
# rho part
input_size_branch_Con_rho = 2
input_size_trunk_rho = 1 # input size of trunk net 
hidden_size_branch_Con_rho = [16, 32, 64]
hidden_size_trunk_rho = [100, 100, 100]
output_size_rho = 100  # outputs neuros

# theta and phi part
input_size_branch_Con_ori = 1
input_size_trunk_ori = 9 # input size of trunk net 
hidden_size_branch_Con_ori = [16, 32, 64]
hidden_size_trunk_ori = [100, 100, 100]
output_size_ori = 100  # outputs neuros

dropout_rate = 0.5
#load trained model
model_path = 'model.pth'

# define the optimizer and loss function
LR = 0.0004 # learning rate
init_LR = LR
BETAS = (0.9, 0.999) # the averge gradient and square of the average gradient
WEIGHT_DECAY = 1e-4

def net_build():
    branch_net_rho = Branch_Net_Con_1D(
        input_size_branch_Con_rho, 
        hidden_size_branch_Con_rho, 
        output_size_rho,
        activation=nn.LeakyReLU,
        kernel_size=11,
        pool_size=2,
        dropout_rate=dropout_rate,
        n_no_pool=2
        ).to(device)
    trunk_net_rho = Trunk_Net(input_size_trunk_rho, hidden_size_trunk_rho, output_size_rho, activation=nn.LeakyReLU).to(device)
    DeepONet_rho = DeepONet(branch_net_rho, trunk_net_rho, num_outputs=1).to(device)

    branch_net_ori = Branch_Net_Con_2D(
        input_size_branch_Con_ori, 
        hidden_size_branch_Con_ori, 
        output_size_ori,
        activation=nn.LeakyReLU,
        kernel_size=3,
        pool_size=2,
        dropout_rate=dropout_rate,
        n_no_pool=2
        ).to(device)
    trunk_net_ori = Trunk_Net(input_size_trunk_ori, hidden_size_trunk_ori, output_size_ori, activation=nn.LeakyReLU).to(device)
    DeepONet_ori = DeepONet(branch_net_ori, trunk_net_ori, num_outputs=1).to(device)

    combined_net = CombineNet(DeepONet1=DeepONet_rho, DeepONet2=DeepONet_ori).to(device)
    return combined_net

############## TRAINING ##############
# run training
def run_training(if_continue):
    datafile = 'dataset_improved_train.npz'
    u_x1, y1, u_x2_x3, y2_y3, G_uy = load_data_from_npz(datafile)
    num = int(len(y1))
    def load_model_to_train():
        checkpoint = torch.load(model_path)
        combined_net.load_state_dict(checkpoint)
        combined_net.train()
        return

    if if_continue:
        load_model_to_train() # load the trained model
    else:
        combined_net.apply(init_weights)


    # Create a dataset and data loader
    batch_size = 512
    G_uy_normalize=(G_uy-G_uy.min())/(G_uy.max()-G_uy.min())

    print(f'Max={G_uy.max()}, Min={G_uy.min()}')

    dataset = TensorDataset(u_x1, y1, u_x2_x3, y2_y3, G_uy_normalize)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    
    num_epochs = 200
    print_every = num / batch_size / 5
    log_file_1 = 'loss_train.txt'
    log_file_2 = 'average_loss_train.txt'
    time_start = time.time()
    log_message=f'{0} {1000}'
    write_log(log_file_1, log_message, 0)
    #run_test('dataset_rho_c1_test.npz', save=True, i=0,max=G_uy.max(),min=G_uy.min())
    for epoch in range(num_epochs):
        running_loss_1 = 0.0
        running_loss_2 = 0.0
        LR = init_LR * (0.95 ** epoch)
        optimizer = optim.Adam(combined_net.parameters(), lr=LR, betas=BETAS, eps=1e-8, weight_decay=WEIGHT_DECAY, amsgrad=False)
        for i, data in enumerate(dataloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            branch_input_rho, trunk_input_rho, branch_input_ori, trunk_input_ori, labels = data
            #branch_input_rho = branch_input_rho.unsqueeze(1)
            branch_input_ori = branch_input_ori.view(batch_size, 1, 30, 30)
            branch_input_rho, trunk_input_rho, branch_input_ori, trunk_input_ori, labels = branch_input_rho.to(device), trunk_input_rho.to(device), branch_input_ori.to(device), trunk_input_ori.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            inputs_rho = (branch_input_rho, trunk_input_rho)
            inputs_ori = (branch_input_ori, trunk_input_ori)
            outputs = combined_net(inputs_rho, inputs_ori)
            
            # Compute the loss
            loss = loss_MSE(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss_1 += loss.item()
            running_loss_2 += loss.item()
            if (i+1) % print_every == 0:  # print every 'print_every' mini-batches
                interval = time.time() - time_start
                log_message = f'{epoch * num + (i + 1) * batch_size} {running_loss_1 / print_every:.8f}'
                write_log(log_file_1, log_message, epoch + i + 1 - print_every)
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss_1 / print_every:.8f}, Time cost:{interval:.2f}')
                running_loss_1 = 0.0
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss_2 / len(dataloader)
        log_message = f'{epoch + 1} {epoch_loss:.8f}'
        write_log(log_file_2, log_message, epoch)
        print(f'Epoch {epoch + 1} Average Loss: {epoch_loss:.8f}')
        running_loss_2 = 0.0
        
        torch.save(combined_net.state_dict(), model_path)
        run_test('dataset_improved_val.npz', save=True, i=epoch,max=G_uy.max(),min=G_uy.min())
        
    interval = time.time() - time_start
    print(f"Training completed!, Time cost: {interval:.2f}")

    #save the trained model
    torch.save(combined_net.state_dict(), model_path)
    print("Model saved!")
    run_test('dataset_improved_test.npz', save=False, i=epoch,max=G_uy.max(),min=G_uy.min())

def run_test(datafile, save, i,max,min):
    def load_model_to_test():
        checkpoint = torch.load(model_path)
        combined_net.load_state_dict(checkpoint)
        combined_net.eval()
        return

    load_model_to_test() # load the trained model

    loss_MSE = nn.MSELoss() # choose mean squared error as the loss function

    u_x1, y1, u_x2_x3, y2_y3, G_uy = load_data_from_npz(datafile)

    # Create a dataset and data loader
    batch_size = 64
    G_uy_normalize=(G_uy-min)/(max-min)

    dataset = TensorDataset(u_x1, y1, u_x2_x3, y2_y3, G_uy_normalize)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)

    # Initialize variables to track test loss
    test_loss = 0.0
    num_batches = len(dataloader)

    # Start testing

    with torch.no_grad():
        for data in dataloader:
            branch_input_rho, trunk_input_rho, branch_input_ori, trunk_input_ori, labels = data
            branch_input_ori = branch_input_ori.view(batch_size, 1, 30, 30)
            branch_input_rho, trunk_input_rho, branch_input_ori, trunk_input_ori, labels = branch_input_rho.to(device), trunk_input_rho.to(device), branch_input_ori.to(device), trunk_input_ori.to(device), labels.to(device)

            # Forward pass
            inputs_rho = (branch_input_rho, trunk_input_rho)
            inputs_ori = (branch_input_ori, trunk_input_ori)
            outputs = combined_net(inputs_rho, inputs_ori)
            
            # Compute the loss
            loss = loss_MSE(outputs, labels)
            test_loss += loss.item()

    # Calculate average test loss
    average_test_loss = test_loss / num_batches
    
    if save:
        log_message = f'{average_test_loss:.8f}'
        write_log('validation_loss.txt', log_message, i)

    print(f"Testing completed! Average Testing Loss: {average_test_loss:.8f}")

def main():
    global combined_net,loss_MSE
    combined_net=net_build()
    optimizer = optim.Adam(combined_net.parameters(), lr=LR, betas=BETAS, eps=1e-8, weight_decay=WEIGHT_DECAY, amsgrad=False) #choose Adam optimizer
    loss_MSE = nn.MSELoss() # choose mean squared error as the loss function
    run_training(if_continue=False)
    run_test('dataset_rho_c1_test.npz',save=False,i=10000,min=-11.9972,max=9.7547)
if __name__ == "__main__":
    main()
