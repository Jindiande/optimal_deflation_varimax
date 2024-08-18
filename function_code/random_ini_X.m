function [X_GT]=random_ini_X(r,n,sparsity)
X_GT=zeros(r,n);
n_round=round(sparsity*n);
for ll = 1:r
   % generate r rows of k-sparse vectors for Y
   X_GT(ll,randperm(n,n_round)) = randn(1,n_round);
end
end