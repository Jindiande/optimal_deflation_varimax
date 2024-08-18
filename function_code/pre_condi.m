function [Y_proj,U,s]=pre_condi(Y,r)
%[U,S,~]=svd(Y);
[U,S,V]=svd(Y);
s=diag(S);
Y_proj=sqrt(size(Y,2))*V(:,1:r)';
end