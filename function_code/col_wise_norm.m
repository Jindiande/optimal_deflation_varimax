
function X=col_wise_norm(X)
   [d1,d2]=size(X);
   X=X./repmat((reshape(sum(X.*X,1),1,d2)).^(1/2),d1,1);
end