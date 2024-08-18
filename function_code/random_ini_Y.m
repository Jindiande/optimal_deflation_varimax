function [Y_noise,Y_GT,X_GT]=random_ini_Y(noise_level,D_base,p,sparsity)
[n,n_sub]=size(D_base);
for ll = 1:p
   % generate p columns of k-sparse vectors for Y
   X_GT(randperm(n_sub,round(sparsity*n_sub)),ll) = randn(round(sparsity*n_sub),1);
end


%fprintf("%d %d\n",size(X_base,1),size(X_base,2))
Y_GT=D_base*X_GT;
Y_noise=Y_GT+noise_level*normrnd(0,1,[n,p]); % noise polluted Y