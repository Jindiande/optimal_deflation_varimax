% This function aims to solve maximise ||AY||_4^{4} A\in O(m,n)
function [error_D,error_X,D_noise,D_noise_no_pre]=L4_MSP(Y,D,X,sparsity,maxstep)% Y should be processed 
[n,p]=size(Y);
[~,n_sub]=size(D);
D_orth_res_tran=MSP(proj_orthogonal_group(randn(n_sub,n)),Y,maxstep);% Transpose of D_orth

[error_D,~,~]=error3(D_noise,D); % find permutation matrix P to solve error  
[error_X,~,~]=error2(X_noise,X);

end