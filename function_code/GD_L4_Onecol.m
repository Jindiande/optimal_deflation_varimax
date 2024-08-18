%Projected Gradient Decent to search for a sparse vector in the problem

% min_q  h_mu(q' * Y),  s.t.  ||q|| =1.
% replacement of TR_Sphere.m 
function q=GD_L4_Onecol(Y,D, q_init,q_exist,MaxIter1,prob_type,if_pgd,if_earlystop,is_modifiedpgd,diagvar)%prob_type=1,3,4
    [p,n] = size(Y);

	% initalize q
    %{
	if nargin > 2,% always use q_init
		q = q_init;
	else
		q = randn(n,1);% random initialization
		q = q / norm(q);
    end
    %}
    %parameter setting 
    %MaxIter1=40; % Gradiant decent ite_num
    %MaxIter2=150; % backtracking line search ite_num
    % if(prob_type==4)
    %     q=q_init;
    % else
    %     q=randn(p,1);
    %     q=q/norm(q);
    % end
    q=q_init;
    alpha=-0.8;
    eta_ini=0.003;
    if(is_modifiedpgd)
        p_full=length(diagvar);
        diagvar=diagvar/n;
        noise_variance=sum(diagvar(p+1:p_full))/(p_full-p);
        diag_modi=diagvar(1:p)-noise_variance*ones([p,1]);
        diag_modi_inv=diag(ones([p,1])./diag_modi);
        noise_diag_modi_inv=noise_variance*diag_modi_inv;
        %noise_diag_modi_inv=noise_variance*10*diag((1./svd(D)).^2)
        noise_diag_modi_inv_wh=noise_diag_modi_inv-norm(noise_diag_modi_inv)*eye(p);
    end
    for iter = 1:MaxIter1
        %U = null(q');
		%[f,g] = l1_exp_approx2(Y,q,mu,true); % evaluate the function, gradient and hessian at q
        %eta = iter^(alpha)*eta_ini;
        if((prob_type~=4)&&size(q_exist,1)~=0)% project gradiant to span of U 
            U=null([q_exist]');
            if(prob_type==1)%
              g=L4_Grad(q,U*U'*Y);
            else%prob_type=3
                g=L4_Grad(q,Y);
                g=project1(U,g);
            end
        else
            g=L4_Grad(q,Y);%prob_type=4
        end
        %{
        if(mod(iter,500)==1)
            fprintf("norm g1=%.20f\n",12*norm((eye(p)-q*q')*(1+q'*noise_diag_modi_inv*q)*noise_diag_modi_inv_wh*q));
            fprintf("norm g2=%.20f\n",norm((eye(p)-q*q')*g));
            fprintf("norm g3=%f\n",norm(g));
            fprintf("norm g4=%f\n",norm((eye(p)-q*q')*L4_Grad(q,D)));
            fprintf("norm g5=%f\n",norm(L4_Grad(q,D)));
            iter=iter;
        end
        %}
        if(if_pgd && is_modifiedpgd)
        	g=g-12*n*(1+q'*noise_diag_modi_inv*q)*noise_diag_modi_inv_wh*q;
            g=(eye(p)-q*q')*g;
           %q=q+eta_ini*iter^(alpha)*g;
           q=q+eta_ini*g;
           q=q/norm(q);
        elseif(if_pgd)
           %q=q+eta_ini*iter^(alpha)*(eye(p)-q*q')*g;
           q=q+eta_ini*(eye(p)-q*q')*g;
           q=q/norm(q);
        elseif(is_modifiedpgd)
           g=g-12*n*(1+q'*noise_diag_modi_inv*q)*noise_diag_modi_inv_wh*q;
           q=g/norm(g);
        else
           q=g/norm(g);
        end
        if(if_earlystop && min(ones(size(D'*q))-abs(D'*q))<0.01)
            break
        end

end
        %{
        if (if_noconstrain)
            g=project1(U,g);
            q=g/norm(g);
        else
            g=project1(U,g);
            q=g/norm(g);
        end
        %}
        %fprintf('%f\n',norm(q));
        %fprintf("Iter is %d,f=%d,norm of grad is %f\n",iter,f_new,norm(g));		    

    %norm((eye(p)-q*q')*12*n*(1+q'*noise_diag_modi_inv*q)*noise_diag_modi_inv_wh*q)
end
