function [Y_cen,mu_r,mu_c,mu]=centering(Y_ori)
        mu_r=Y_ori*ones(size(Y_ori,2),1);%column vector
        mu_c=ones(1,size(Y_ori,1))*Y_ori;%row vector
        mu=ones(1,size(Y_ori,1))*Y_ori*ones(size(Y_ori,2),1);
        Y_cen=Y_ori-mu_r*ones(1,size(Y_ori,2))-ones(size(Y_ori,1),1)*mu_c+ones(size(Y_ori,1),1)*mu*ones(1,size(Y_ori,2));
end