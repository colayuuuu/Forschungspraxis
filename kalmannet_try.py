import torch
import torch.nn as nn

class KalmanNetNN(torch.nn.Module):
     def __init__(self):
         super().__init__()
         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


     def Build(self,ssModel):
         self.SystemDynamics(ssModel.F,ssModel.H)
     def SystemDynamics(self,trans_F,obser_H):
         self.m = self.trans_F.size()[0]
         self.n=self.obser_H.size()[0]

     def KalmanGain(self):
         D_in=self.m+self.n
         D_out=self.m*self.n
         self_first_linear = torch.nn.Linear(D_in,H1)
         self_first.act=torch.nn.ReLU()
         # H1 is the output of the fully connected linear layer while the input of the GRU
         ##GRU
         self.input_dim = H1
         self.hidden_dim=self.m**2+seld.n**2
         self.n_layers =1
         self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)
         self.second_linear= torch.nn.Linear(self.hidden_dim,D_out)
         self.second_act=torch.nn.ReLU()

     def init_state(self,mx):
         self.mx_prior=mx.to(self.device,non_blocking = True)
         self.mx_posterior = mx.to(self.device, non_blocking=True)
         self.mx_process_posterior_0 = mx.to(self.device, non_blocking=True)

     def pridict_step(self):
        self.x_process_priori_0=torch.matmul(self.trans_F,self.x_process_posterior_0)
        self.y_process_0=torch.matmul(self.obser_H,self.x_process_priori_0)
        self.x_prev_prior = self.x_prior
        self.x_prior = torch.matmul(self.trans_F, self.x_posterior)
        self.m1y = torch.matmul(self.obser_H , self.x_prior)

     def estimation_KalmanGain(self,y):
        #input feature F2 & F4
        #y is the input measurement
        # F2-feature
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)
        #F4-feature
        dm1x = self.x_posterior - self.x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)
        KG = self.KGain_step(KGainNet_in)
        self.KGain = torch.reshape(KG,(self.m,self.n))


     def KNet_step(self, y):
           # Compute Priors
           self.step_prior()

           # Compute Kalman Gain
           self.estimation_KalmanGain(y)
           y_obs = torch.unsqueeze(y, 1)
           dy = y_obs - self.m1y

            # Compute the 1-st posterior moment
           INOV = torch.matmul(self.KGain, dy)
           self.m1x_posterior = self.m1x_prior + INOV
           return torch.squeeze(self.m1x_posterior)

     def forward(self, yt):
            yt = yt.to(self.device,non_blocking = True)
            return self.KNet_step(yt)


