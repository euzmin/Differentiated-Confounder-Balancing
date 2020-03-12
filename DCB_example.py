import numpy as np
import torch
import DCB_model as Model


def loadData(filePath):
    Data = []
    with open(filePath, 'r') as f:
        for line in f.readlines():
            line_arr = line.strip().split()
            float_list = map(float, line_arr)
            Data.append(list(float_list))

    return Data


def Y_generator(p_f, T_f, X_f):
    Normal_0_3 = torch.randn([m, 1]) * (3 ** 0.5)
    Y_f = T_f + Normal_0_3
    for j in range(p_f):
        weight = 0
        if (j + 1) % 2 == 0:
            weight = (j + 1) // 2
        Y_f = Y_f + weight * X_f[:, j].unsqueeze(-1) + X_f[:, j].unsqueeze(-1).mul(T_f)
    return Y_f


def DCB_with_setting_parameters(x_t, y_t, x_c, y_c):
    lambda0 = 10.0
    lambda1 = 10.0
    lambda2 = 100.0
    lambda3 = 1.0
    lambda4 = 0.1
    MAXITER = 1000
    ABSTOL = 1e-6
    model = Model.DCB_model(x_t, y_t, x_c, y_c)
    ATE, W, beta = model.DCB(lambda0, lambda1, lambda2, lambda3, lambda4, MAXITER, ABSTOL)
    W = W.mul(W)
    ATE_residual = torch.mean(y_t) - torch.mean(x_t, 0).unsqueeze(0).mm(beta) + W.permute(1, 0).mm(y_c - x_c.mm(beta))
    return W, ATE, ATE_residual

# x[2000,50],T_logit,Y_linear,s_c=1,s_r=0.2


if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    torch.set_flush_denormal(False)
    m = int(2000)
    p = int(50)

    sc = 1
    rc = 0.2

    #  dimension of T
    p_t = int(p * rc)
    ATT_error_list = None
    for example_iter in range(100):
        X = torch.randn([m, p])

        # T服从伯努利分布，因此根据sigmoid函数，划分T
        T = X[:, :p_t].mm(torch.ones([p_t, 1])) * sc + torch.randn([m, 1])
        T[T > 0] = 1
        T[T < 0] = 0

        Y = Y_generator(p, T, X)
        Y1 = Y_generator(p, 1, X)
        Y0 = Y_generator(p, 0, X)
        #  ground truth
        ATT_gt = torch.mean(Y1[T == 1] - Y0[T == 1])

        X_t = X[T.squeeze(-1) == 1, :]
        X_c = X[T.squeeze(-1) == 0, :]

        Y_t = Y[T.squeeze(-1) == 1, :]
        Y_c = Y[T.squeeze(-1) == 0, :]

        m_X_t = list(X_t.size())[0]
        m_X_c = list(X_c.size())[0]

        ATE_naive = torch.mean(Y_t) - torch.mean(Y_c)
        # print(f'ATE_naive : {ATE_naive}')
        W_dcb, ATE_dcb, ATE_dcb_regression = DCB_with_setting_parameters(X_t, Y_t, X_c, Y_c)
        ATT_error = torch.Tensor([ATE_naive, ATE_dcb, ATE_dcb_regression]) - ATT_gt
        if ATT_error_list is None:
            ATT_error_list = ATT_error.unsqueeze(0)
        else:
            ATT_error_list = torch.cat([ATT_error_list, ATT_error.unsqueeze(0)], 0)

    error_mean = ATT_error_list.mean(dim=0)
    error_std = ATT_error_list.std(dim=0)
    error_mae = ATT_error_list.abs().std(dim=0)
    error_mae = torch.mean(torch.abs(torch.Tensor(ATT_error_list)), 0)
    error_rmse = torch.sqrt(torch.mean(torch.Tensor(ATT_error_list) ** 2, 0))

    print('ATE_naive, ATE_dcb, ATE_dcb_regression')
    print(f'Bias: {error_mean}')
    print(f'SD  : {error_std}')
    print(f'MAE : {error_mae}')
    print(f'RMSE: {error_rmse}')




