import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Helper functions for SL Decomposition
def So(tau, X):
    Y = X.abs()-tau
    Y[Y<0] = 0
    r = torch.sign(X) * Y
    return r

def Do(tau, X):
    in_device = X.device
    (U,S,V) = torch.svd(X)
    
    Sbatch = torch.zeros((S.shape[1],S.shape[1]))
    for nB in range(S.shape[1]):
        Sbatch[nB,nB] = S[0,nB]
    U = U.to(in_device)
    Sbatch = Sbatch.to(in_device)
    V = V.to(in_device)
    r = torch.matmul(torch.matmul(U, So(tau, Sbatch)), V.permute(0, 2, 1))
    return r


def SLDecomposition(X, lam=None, mu=None, maxIter=10, plot=False):
    full_shape = X.shape
    Xtype = X.type()

    shape = X.shape[-2:]
    X = X.view(full_shape[0],full_shape[1],-1).permute(0,2,1).float()

    if lam is None:
        lam = 0.05*1.0/torch.tensor(X.shape).float().max().sqrt()
    if mu is None:
        mu = 10*lam
    
    L = torch.zeros_like(X)
    S = torch.zeros_like(X)
    Y = torch.zeros_like(X)

    normX = torch.norm(X, 'fro')
    errors = []
    err = 10000
    
    if plot:
        plt.ion()
        plt.figure()

    end = "\r"
    for it in range(maxIter):
        L = Do(1.0/mu, X - S + (1.0/mu)*Y)
        L = F.relu(L)
        S = So(lam/mu, X - L + (1.0/mu)*Y)
        S = F.relu(S)

        Z = X - L - S
        Y = Y + mu*Z
        last_err = err
        err = torch.norm(Z, 'fro') / normX
        
        errors.append(err.item())
        # if err>last_err:
            # print('last lambdaR: ' + str(lam))
            # break


        if plot:
            plt.clf()
            plt.subplot(2,4,1)
            plt.imshow( X[0,:,-1].view(*shape).detach().cpu().numpy() )
            plt.subplot(2,4,2)
            plt.imshow( L[0,:,-1].float().view(*shape).detach().cpu().numpy() )
            plt.subplot(2,4,3)
            plt.imshow( S[0,:,-1].float().view(*shape).detach().cpu().numpy() )
            plt.subplot(2,4,4)
            plt.imshow( Y[0,:,-1].float().view(*shape).detach().cpu().numpy() )
            plt.subplot(2,1,2);
            plt.plot(list(range(0,it+1)), errors)
            plt.pause(1)
            plt.draw()

        if it==maxIter-1:
            end = "\n"
        print(str(it) + ' / ' + str(maxIter) + 'th iteration.  Error: \t' + str(err.item()), end=end)
    
    S = S.permute(0,2,1).view(*full_shape).type(Xtype)
    L = L.permute(0,2,1).view(*full_shape).type(Xtype)
    Y = Y.permute(0,2,1).view(*full_shape).type(Xtype)
    return L,S,Y



class SLNet(nn.Module):
    def __init__(self, n_temporal_frames=1, use_bias=False, mu_sum_constraint=5e-3, alpha_l1=10.0):
        super(SLNet, self).__init__()
        self.n_frames = n_temporal_frames
        self.mu_sum_constraint = torch.tensor(float(mu_sum_constraint))
        self.alpha_l1 = alpha_l1

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(n_temporal_frames, n_temporal_frames*2, 1, 1, 0, bias=use_bias)
        self.conv2 = nn.Conv2d(n_temporal_frames*2, n_temporal_frames*1, 1, 1, 0, bias=use_bias)

    def forward(self, input):
        x = (self.conv1(input))
        x = self.relu(self.conv2(x))

        return x