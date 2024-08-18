function D_noise_orth_inv_trans=ortho_init(Y,A_gt,maxstep)
    [p,n]=size(Y);
    [~,r]=size(A_gt);
    %error_each_r=zeros(r,1)
    %[~,r]=size(D);
    time_try=r;
    end_colum=p-1;
    D_rec
    for i=1:time_try
        if size(D_noise_orth_inv_trans,2)~=0
             q_init=null(D_rec')*randn(p-size(D_rec,2),1);% q_init lies in null space of D_noise
             q_init=q_init/norm(q_init);
        else
             q_init=randn(p,1);
             q_init=q_init/norm(q_init);
        end

        q = L4_onecolumn(q_ini,Y,maxstep); % using algorithm to generate q
        if(min(ones(size(D_noise_orth_inv_trans'*q))-abs(D_noise_orth_inv_trans'*q))<10^(-1))
            continue;
        end
            
        if size(D_noise_orth_inv_trans,2)==end_colum% D_noise_orth_inv_trans is n by n-1
            q=null(transpose(D_noise_orth_inv_trans));
        end  

        D_noise_orth_inv_trans=[D_noise_orth_inv_trans,q];
        if(size(D_noise_orth_inv_trans,2)==r)
            break;
        end
    end
end