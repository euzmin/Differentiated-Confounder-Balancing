import numpy as np
import torch


class DCB_model(object):
    def __init__(self, X_t, Y_t, X_c, Y_c):

        self.X_t = X_t
        self.X_c = X_c
        self.Y_t = Y_t
        self.Y_c = Y_c
        # get dimension of X_t and X_c

        self.n_t = list(X_t.size())[0]
        self.n_c = list(X_c.size())[0]

        self.p = list(X_t.size())[1]
        # print(f'the size of mean xt is :{torch.mean(X_t, 0).size()}')
        self.mean_X_t = torch.mean(X_t, 0).unsqueeze(-1)  # 求xt平均值，为优化w做准备
        # initialize parameters

        self.W = torch.ones([self.n_c, 1])/self.n_c
        self.W_pre = self.W
        self.beta = torch.ones([self.p, 1])# /self.p
        self.beta_pre = self.beta
        self.parameter_iter = 0.5
        self.lambda_W = 1
        self.lambda_beta = 1

    def DCB(self, p_lambda0, p_lambda, p_delta, p_mu, p_v, MAXITER, ABSTOL):
        J_loss = torch.ones([MAXITER, 1]) * -1

        for Iter in range(MAXITER):
            # print(f'iter: {Iter}')
            alpha = (Iter + 1) / (Iter + 4)
            # beta

            # update grad
            y = self.beta
            self.beta = self.beta + alpha * (self.beta - self.beta_pre)
            f_base=self.loss_function(self.W,self.beta,p_lambda0,p_lambda,p_delta,p_mu)

            # find grad
            while True:
                diff_X = self.getDiff_X_t_c()
                grad_lambda0 = 2 * p_lambda0 * self.beta.permute(1,0).mm(diff_X)*diff_X
                grad_lambda=-2*p_lambda*self.X_c.permute(1,0).mm((1+self.W.mul(self.W)).mul(self.Y_c-self.X_c.mm(self.beta)))
                grad_delta = 2 * p_mu * self.beta
                grad_beta = grad_lambda0 + grad_lambda + grad_delta
                # print(f'beta grad_beta:{grad_beta.size()}')
                z = self.prox_l1(self.beta - self.lambda_beta * grad_beta, self.lambda_beta * p_v)
                # print(f'beta z:{z.size()}')
                f_x = self.loss_function(self.W, z, p_lambda0, p_lambda, p_delta, p_mu)
                #f_obj=f_base+grad_beta.permute(1,0).mm(self.beta-z)+(1/(2*self.lambda_beta))*torch.sum((z-self.beta)**2)
                f_obj=f_base+grad_beta.permute(1,0).mm(z-self.beta)+(1/(2*self.lambda_beta))*torch.sum((z-self.beta)**2)

                if f_x <= f_obj:  # ?
                    # print(f'beta: f_base:{f_base}, f_x:{f_x}, f_obj:{f_obj}')
                    break
                self.lambda_beta = self.parameter_iter * self.lambda_beta

            self.beta_pre = y
            self.beta = z

            # W

            # update grad
            y = self.W
            self.W = self.W + alpha * (self.W - self.W_pre)
            f_base=self.loss_function(self.W,self.beta,p_lambda0,p_lambda,p_delta,p_mu)
            # print(f'W : f_base: {f_base}')
            # find grad
            W_times = 0
            while True:
                diff_X = self.getDiff_X_t_c()
                grad_lambda0 = -4 * p_lambda0 * self.beta.permute(1,0).mm(diff_X)*self.X_c.mm(self.beta).mul(self.W)
                grad_lambda = 2 * p_lambda * ((self.Y_c - self.X_c.mm(self.beta))**2).mul(self.W)
                grad_delta = 4 * p_delta * self.W.mul(self.W).mul(self.W)
                grad_W = grad_lambda0 + grad_lambda + grad_delta
                # print(f'W grad_W:{grad_W.size()}')
                z = self.prox_l1(self.W - self.lambda_W * grad_W, 0)
                # print(f'W z:{z.size()}')
                f_x = self.loss_function(z,self.beta,p_lambda0, p_lambda, p_delta, p_mu)
                #f_obj=f_base+grad_W.permute(1,0).mm(self.W-z)+(1/(2*self.lambda_W))*torch.sum((z-self.W)**2)

                f_obj=f_base+grad_W.permute(1,0).mm(z-self.W)+(1/(2*self.lambda_W))*torch.sum((z-self.W)**2)

                if f_x <= f_obj:
                    # print(f'W: f_base:{f_base}, f_x: {f_x}, f_obj: {f_obj}')
                    break
                else:
                    W_times += 1
                   # print(W_times)
                self.lambda_W = self.parameter_iter * self.lambda_W

            self.W_pre = y
            self.W = z

            self.W = self.W/torch.sqrt(torch.sum(self.W.mul(self.W)))

            ATT = torch.mean(self.Y_t) - (self.W.mul(self.W)).permute(1,0).mm(self.Y_c)
            loss_val=self.loss_function(self.W,self.beta,p_lambda0,p_lambda,p_delta,p_mu)
            J_loss[Iter] = loss_val + p_v * torch.sum(abs(self.beta))
            # print(f'Iter:{Iter}, w:{self.W}, beta:{self.beta}')
            grad_lambda0_A = self.beta.mul(self.getDiff_X_t_c())
            grad_lambda0 = p_lambda0 * grad_lambda0_A.permute(1,0).mm(grad_lambda0_A)
            grad_lambda = p_lambda * torch.sum((self.Y_c - self.X_c.mm(self.beta)) ** 2)
            grad_mu_A = p_mu * ((self.W.mul(self.W)).permute(1,0).mm(self.W.mul(self.W)))
            grad_mu_B = p_v * torch.sum(self.beta ** 2)
            grad_mu = grad_mu_A + grad_mu_B
            grad_v = p_v * torch.sum(abs(self.beta))
            if (Iter > 0 and abs(J_loss[Iter] - J_loss[Iter - 1]) < ABSTOL) or Iter == MAXITER - 1:
                if Iter != MAXITER - 1:
                    print(f'Get the optimal results at iteration {Iter}')
                print(f'our iteration {Iter} ... J_error: {J_loss[Iter]}, lambda0: {grad_lambda0}')
                print(f'lambda: {grad_lambda}, mu: {grad_mu}, v: {grad_v}')
                break
        return ATT, self.W, self.beta

    def loss_function(self, W, beta,  p_lambda0, p_lambda1, p_delta, p_mu):
        obj_part = beta.permute(1,0).mm(self.mean_X_t - self.X_c.permute(1, 0).mm(W.mul(W)))
        lambda0_part = p_lambda0 * obj_part.permute(1,0).mm(obj_part)
        lambda_part_A = (1+W.mul(W)).permute(1,0)
        # print(f'lambda_part_A:{lambda_part_A.size()}')
        lambda_part_B = (self.Y_c - self.X_c.mm(beta))**2
        # print(f'lambda_part_B:{lambda_part_B.size()}')
        lambda_part = p_lambda1 * lambda_part_A.mm(lambda_part_B)
        delta_part = p_delta * ((W.mul(W)).permute(1,0)).mm(W.mul(W))
        mu_part = p_mu * torch.sum(beta**2)
        # print(f'loss_function: lambda0_part:{lambda0_part},lambda_part:{lambda_part},delta_part:{delta_part},mu_part:{mu_part}')
        ans = lambda0_part + lambda_part + delta_part + mu_part
        # print(f'loss_function: ans:{ans}')
        return ans

    def prox_l1(self, v, p_lambda):
        x = v - p_lambda
        y = -v - p_lambda
        x[x < 0] = 0
        y[y< 0] = 0
        # print(f'prox_l1 x:{(x-y).size()}')
        return x-y

    def getDiff_X_t_c(self):
        return self.mean_X_t - self.X_c.permute(1, 0).mm(self.W.mul(self.W))
