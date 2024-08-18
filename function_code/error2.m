%using cvx minimise (1/2)*|X-P*X_groud_truth|_2 
%given X and X_groud_truth, where P
% is permutation matrix solved with SOPT3
function [error_1,P_out,P]=error2(X,X_groud_truth)
    n1=size(X,1);
    n2=size(X,2);
	cvx_quiet(1);
	cvx_begin 
		variable P(n1,n1)  ;
		minimize(norm(X-P*X_groud_truth,'fro'))%+norm(P,1))
		subject to
        (P)*ones(n1,1)<=ones(n1,1)
        (P)*ones(n1,1)>=-1*ones(n1,1)
        ones(1,n1)*(P)<=ones(1,n1)
        ones(1,n1)*(P)>=-1*ones(1,n1)
        %A.*P==0
	cvx_end
    % solve  linear assignemnt problem by matchpairs function
    P_extend=repmat(P,[1,1,n1]);
    BasisPosi=repmat(reshape(transpose(eye(n1,n1)),[1,n1,n1]),[n1,1,1]);
    BasisNega=repmat(reshape(-1*transpose(eye(n1,n1)),[1,n1,n1]),[n1,1,1]);
    LossPoss=reshape(vecnorm(P_extend-BasisPosi,2,2),[n1,n1]) ;
    LossNega=reshape(vecnorm(P_extend-BasisNega,2,2),[n1,n1]) ;
    Sign=2*double(LossPoss<LossNega)-ones(n1,n1);
    Loss=min(LossPoss,LossNega);
    P_out=zeros(n1,n1);
    index=matchpairs(Loss,1000);
    ind=sub2ind([n1,n1],index(:,1),index(:,2));
    P_out(ind)=1;
    P_out=Sign.*P_out;
    error_1=(1/2)*norm(X-P_out*X_groud_truth,'fro');
     
end