%Implementation of algorithm 2 provided in 'Aubby and Yuan, 2023'.
function q=initialization_ica(Y, is_dataspliting, repeat_times)
[r,n]=size(Y);
if(is_dataspliting)
%1+1
    Y1=Y(:,1:round(n/2));
    Y2=Y(:,round(n/2)+1:n);
    [~,n1]=size(Y1);
    [~,n2]=size(Y2);
    Y_cov=pagemtimes(reshape(Y1,[r,1,n1]),reshape(Y1,[1,r,n1]));%r\times r\times n
    Y_cov_vec=reshape(Y_cov,[r*r,n1]);%vec operation
    vec_I=reshape(eye(r,r),[r*r,1]);
    barY=mean(Y_cov_vec,2);
    M1_hat=(1/n1)*(Y_cov_vec-barY)*(Y_cov_vec-barY)'+vec_I*vec_I'; 
    M_0=calculate_M0(r);
    M1=M1_hat-reshape(M_0,[r^2,r^2]);
    [U1,~,~]=svd(M1);
    P_M1=U1*U1';

    Y_cov=pagemtimes(reshape(Y2,[r,1,n2]),reshape(Y2,[1,r,n2]));%r\times r\times n
    Y_cov_vec=reshape(Y_cov,[r*r,n2]);%vec operation
    vec_I=reshape(eye(r,r),[r*r,1]);
    barY=mean(Y_cov_vec,2);
    M2_hat=(1/n2)*(Y_cov_vec-barY)*(Y_cov_vec-barY)'+vec_I*vec_I'; 
    M2=M2_hat-reshape(M_0,[r^2,r^2]);
    M2=reshape(P_M1*M2*P_M1,[r,r,r,r]);
    G=randn([repeat_times,r,r]);
    M_I=tensorprod(M2,G,[3,4],[2,3]);%[r, r repeat_times]
    [U,S]=pagesvd(M_I);
    [~,I]=max(S(1,1,:));%I=index of maximal sing value
    q=U(:,1,I);
else
    % barY=reshape((1/n)*Y*Y',[r*r,1]);%r^2
    % barY=barY
    Y_cov=pagemtimes(reshape(Y,[r,1,n]),reshape(Y,[1,r,n]));%r\times r\times n
    Y_cov_vec=reshape(Y_cov,[r*r,n]);%vec operation
    vec_I=reshape(eye(r,r),[r*r,1]);
    barY=mean(Y_cov_vec,2);
    M_hat=(1/n)*(Y_cov_vec-barY)*(Y_cov_vec-barY)'+vec_I*vec_I'; 
    M_0=calculate_M0(r);
    M=M_hat-reshape(M_0,[r^2,r^2]);
    [U,~,~]=svd(M);
    P_M=U*U';
    M=reshape(P_M*M*P_M,[r,r,r,r]);
    G=randn([repeat_times,r,r]);
    M_I=tensorprod(M,G,[3,4],[2,3]);%[r, r repeat_times]
    [U,S]=pagesvd(M_I);
    [~,I]=max(S(1,1,:));%I=index of maximal sing value
    q=U(:,1,I);
end

end
