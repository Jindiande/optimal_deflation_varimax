function I=patches_to_image(Y)
[p,n]=size(Y);
I=zeros(sqrt(p*n),sqrt(p*n));
k=1;
for row_index=1:sqrt(n)
    for col_index=1:sqrt(n)
        I((row_index-1)*sqrt(p)+1:row_index*sqrt(p),(col_index-1)*sqrt(p)+1:col_index*sqrt(p))=reshape(Y(:,k),[sqrt(p),sqrt(p)]);
        k=k+1;
    end
end