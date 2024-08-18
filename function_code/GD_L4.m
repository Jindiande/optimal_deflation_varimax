
function [D_noise_orth_inv_trans]=GD_L4(Y,A_gt,MaxIter1,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,diagvar,repeat_times)% Y should be processed  
[p,n]=size(Y);
[~,r]=size(A_gt);
%error_each_r=zeros(r,1)
%[~,r]=size(D);
D_noise_orth_inv_trans=[];% left inverse of dictionry
D_pertubed=[];
timetry=10^3;
if(if_inisvd)
    [U,~,~]=svd(Y);
    U=U(1:r);
end

%Y_noise_orth=real(inv(sqrtm(Y_noise*transpose(Y_noise)/(p*sparsity)))*Y_noise)  %  preconditioning for Y_noise
for time=1:timetry% each time generate a column of left inverse of dictionary
    %fprintf("i=%d,k=%d,time=%d,column=%d,real column%d\n",i,k,time,column,size(D_noise_orth_inv_trans,2))
    %{
    if (size(D_noise_orth_inv_trans,2)==(end_colum-1))% D_noise_orth_inv_trans is p by p-1
        q=null(transpose(D_noise_orth_inv_trans));
        D_noise_orth_inv_trans=[D_noise_orth_inv_trans,q(:,1)];
        break
    end  
    %}
    q_init=randn(p,1);%
    % z=q_init'*Y;
    % [~,I]=max(sum(z.*z.*z.*z,2));
    % q_init=q_init(:,I);
    %{
    if(size(D_noise_orth_inv_trans,2)==r)
        q=null(D_noise_orth_inv_trans');
        D_noise_orth_inv_trans=[D_noise_orth_inv_trans,q];
        break
    end
    %}
    %{
    if (time~=1)     
         %q_init=(eye(p)-D_noise_orth_inv_trans*D_noise_orth_inv_trans')*randn(p,1);
         Nspace=null(D_noise_orth_inv_trans');
         q_init=Nspace*Nspace'*q_init;% q_init lies in null space of D_noise
         q_init=q_init/norm(q_init);
    else
         q_init=q_init/norm(q_init);
    end
    %}
    if(time==1)
        q = GD_L4_Onecol(Y,A_gt,q_init,D_noise_orth_inv_trans,MaxIter1,prob_type,if_pgd,if_earlystop,is_modifiedpgd,diagvar); % using algorithm to generate q
        if( if_pertub_full | if_pertub_onec)
            q_t=q+perturb_rate*randn(p,1);
        else
            q_t=q;
        end
    else
        if(~if_tensor_ini)
            if(if_nullspace)%prob_type=4
                %fprintf("test\n");
                 Nspace=null(D_pertubed');
                 %q_init=Y*ones(n,1);
                 %q_init=q_init/norm(q_init);
                 if(if_inisvd)
                      q_init=U*U'*q_init;% q_init lies in null space of D_noise
                      q_init=q_init/norm(q_init);
                 end
                 q_init=Nspace*Nspace'*randn(p,repeat_times);
                 q_init=q_init ./ repmat(sqrt(sum(q_init.^2)),size(q_init,1),1);
                 z=q_init'*Y;
                [~,I]=max(sum(z.*z.*z.*z,2));
                 q_init=q_init(:,I);
                 q_init=q_init/norm(q_init);
                 %{
                 [q_init,~,~]=svd((eye(p)-D_pertubed*D_pertubed')*U);
                 q_init=q_init(:,1:r-size(D_pertubed,2))*randn(r-size(D_pertubed,2),1);
                 q_init=q_init/norm(q_init);
                 %}
            % else
            %          q_init=(eye(p)-D_pertubed*D_pertubed')*randn(p,1);
            %          q_init=q_init/norm(q_init);
            elseif (if_mrs)%mrs
                Nspace=null(D_pertubed');
                nullspace=Nspace*Nspace';
                if(~is_inidenoise)%no improvement
                    noise_diag_modi_inv=zeros([p,1]);
                else%improvement
                    p_full=length(diagvar);
                    diagvar=diagvar/n;
                    noise_variance=sum(diagvar(p+1:p_full))/(p_full-p);
                    diag_modi=diagvar(1:p)-noise_variance*ones([p,1]);
                    diag_modi_inv=diag(ones([p,1])./diag_modi);
                    noise_diag_modi_inv=noise_variance*diag_modi_inv;
                    %noise_diag_modi_inv=noise_variance*10*diag((1./svd(D)).^2)
                end
                q_init=initialization_MRS(Y, nullspace,if_ini_spliting, repeat_times, noise_diag_modi_inv);
                % fprintf("inner product=%d, size=%d, ",max(abs(q_init'*D_pertubed)), size(Nspace,2));              
            end
        else% tensor_initlization, prob_type must be 1
            Nspace=null([D_pertubed]');
            q_init=initialization_ica(Nspace*Nspace'*Y, if_ini_spliting, repeat_times)
        end
        q = GD_L4_Onecol(Y,A_gt,q_init,D_pertubed,MaxIter1,prob_type,if_pgd,if_earlystop,is_modifiedpgd,diagvar); % using algorithm to generate q
        if(if_pertub_full)
            q_t=q+perturb_rate*randn(p,1);
        else
            q_t=q;
        end
        
        if(if_checkrepeat && max(abs(D_noise_orth_inv_trans'*q))>0.7)
           continue;
        end
        %}
        %max(abs(D_pertubed'*q))
        %[v,i]=max(abs(A_gt'*q))
    end
    % [err,I]=min(ones(size(A_gt'*q))-abs(A_gt'*q));
    % fprintf("err=%d,index=%d\n",err,I);
    D_noise_orth_inv_trans=[D_noise_orth_inv_trans,q];
    D_pertubed=[D_pertubed,q_t];
    %{
    if(size(D_noise_orth_inv_trans,2)==r-1)
        Nspace=null(D_noise_orth_inv_trans');
        D_noise_orth_inv_trans=[D_noise_orth_inv_trans,Nspace(:,1)];
        break
    end
    %}
    
    if(size(D_noise_orth_inv_trans,2)==r)
       break
    end
    %}
                %fprintf("\n");
end

if(if_checkrepeat && size(D_noise_orth_inv_trans,2)~=r)
    Nspace=null(D_noise_orth_inv_trans');
    D_noise_orth_inv_trans=[D_noise_orth_inv_trans,Nspace(:,1:r-size(D_noise_orth_inv_trans,2))];
end
%D_noise_orth_inv_trans=[q_1,D_noise_orth_inv_trans(:,2:r)];

%rank(D_noise_orth_inv_trans)
end


%[error_D,~,~]=error3(D_noise_orth_inv_trans,D); 
%[error_X,~,~]=error2(D_noise_orth_inv_trans'*Y,X);