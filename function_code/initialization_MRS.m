function q=initialization_MRS(Y, nullspace,is_dataspliting, repeat_times,diagvar_noise)
    %is_dataspliting=false;
    [r,n]=size(Y);
    diagvar_noise=repmat(diagvar_noise,[1,1,repeat_times]);%r\times r\times repeat_times
    diagvar_noise=diagvar_noise+repmat(eye(r,r),[1,1,repeat_times]);
    if is_dataspliting
        G=randn(r,r,repeat_times);
        Y_1=Y(:,1:round(n/2));
        Y_2=Y(:,round(n/2)+1:n);
        Y_3d_1=repmat(Y_1,[1,1,repeat_times]);%r\times n\times repeat_times
        sig_1=pagemtimes(pagemtimes(permute(Y_3d_1,[2,1,3]),G), Y_3d_1);%n\times n\times repeat_times
        I_half=repmat(eye(round(n/2),round(n/2)),[1,1,repeat_times]);%
        diag_1=bsxfun(@times, sig_1, I_half);
        diag_1=reshape(diag_1,[round(n/2),round(n/2),repeat_times]);
        M_hat_1=pagemtimes(pagemtimes(Y_3d_1,diag_1),permute(Y_3d_1,[2,1,3]));%r\times r\times repeat_time
        M_hat_1=M_hat_1-pagemtimes(pagemtimes(diagvar_noise,(G+permute(G,[2,1,3]))),diagvar_noise);
        I=repmat(eye(r,r),[1,1,repeat_times]);
        G_times_N=pagemtimes(G,diagvar_noise);%r\times r\times repeat_times
        tracesum=sum(bsxfun(@times, G_times_N, I),[1,2]);%1\times 1\times repeat_times
        M_hat_1=M_hat_1-bsxfun(@times, tracesum, G);

        Y_3d_2=repmat(Y_2,[1,1,repeat_times]);%r\times n\times repeat_times
        sig_2=pagemtimes(pagemtimes(permute(Y_3d_2,[2,1,3]),G), Y_3d_2);%n\times n\times repeat_times
        diag_2=bsxfun(@times, sig_2, I_half);
        diag_2=reshape(diag_2,[round(n/2),round(n/2),repeat_times]);
        M_hat_2=pagemtimes(pagemtimes(Y_3d_2,diag_2),permute(Y_3d_2,[2,1,3]));%r\times r\times repeat_times
        [V_2,D]=pagesvd(M_hat_2);
        V_2=pagemtimes(V_2,permute(V_2,[2,1,3]));
        M_hat=pagemtimes(pagemtimes(V_2,M_hat_1),V_2);

        nullspace=repmat(nullspace,[1,1,repeat_times]);
        [V,D]=pagesvd(pagemtimes(pagemtimes(nullspace,M_hat),nullspace));
        [~,I]=max(D(1,1,:)-D(2,1,:));
        q=V(:,1,I);     
    else
        G=randn(r,r,repeat_times);
        Y_3d=repmat(Y,[1,1,repeat_times]);%r\times n\times repeat_times
        sig=pagemtimes(pagemtimes(permute(Y_3d,[2,1,3]),G), Y_3d);%n\times n\times repeat_times
        I=repmat(eye(n,n),[1,1,repeat_times]);%
        diag=bsxfun(@times, sig, I);
        diag=reshape(diag,[n,n,repeat_times]);
        M_hat=(n/3)*pagemtimes(pagemtimes(Y_3d,diag),permute(Y_3d,[2,1,3]));%r\times r\times repeat_times
        M_hat=M_hat-(G+permute(G,[2,1,3]));
        %{
        M_hat=M_hat-pagemtimes(pagemtimes(diagvar_noise,(G+permute(G,[2,1,3]))),diagvar_noise);
        I=repmat(eye(r,r),[1,1,repeat_times]);
        G_times_N=pagemtimes(G,diagvar_noise);%r\times r\times repeat_times
        tracesum=sum(bsxfun(@times, G_times_N, I),[1,2]);%1\times 1\times repeat_times
        M_hat=M_hat-bsxfun(@times, tracesum, G);
        %}
        nullspace=repmat(nullspace,[1,1,repeat_times]);
        [V,D]=pagesvd(pagemtimes(pagemtimes(nullspace,M_hat),nullspace));
        [~,I]=max(D(1,1,:)-D(2,2,:));
        q=V(:,1,I);
        %fprintf("test\n");
    end



end