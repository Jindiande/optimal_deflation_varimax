clear all; clc;
addpath('../function_code');
p=[20,30,50,100,200,300];
r=[5,10,15,20,25,30];
theta=0.1;
noise_signal_ratio=[0:0.2:1];
alpha_log=[-1:0.1:0];
TOL = 1e-5;
tau=0.1;
MAX_ITER=10^5;
noise_cata={'version_1'};
method_cata={'PCA-dVarimax', 'PCA-dVarimax-1','PCA-dVarimax-2', 'PCA-dVarimax-3'};
%n=1.2*10^(4);
n=[50,100,200,500,700,1000,3000,5000];
prob_type=4;
if_nullspace=true;
if_pgd=false;
if_pertub_onec=false;
if_pertub_full=false;
if_inisvd=false;
if_earlystop=false;
if_checkrepeat=false;
if_centering=false;
is_modifiedpgd=false;

if_checkrepeat=false;
repeat_time_ini=1;
if_ini_spliting=false;
if_tensor_ini=false;
if_mrs=true;
if_nullspace=false;
is_inidenoise=true;
repeat_time_ini=100;
prob_type=4;

repeat_time1=50;
repeat_time2=1;
if (if_pgd)
    maxstep=5*10^3;
else
    maxstep=60;
end
perturb_rate=0;
n_default_index=4;
p_default_index=1;
r_default_index=1;
eps_default_index=4;

error_p=zeros(length(method_cata),length(noise_cata),length(p));
error_eps=zeros(length(method_cata),length(noise_cata),length(noise_signal_ratio));

for cata_index=1:length(noise_cata)
    for p_index=1:length(p)
            error_L4_sum=0;
            error_L4_sum_before=0;
            error_L4_sum_after=0;
            error_L4_sum_final=0;
            fprintf("p_index=%d\n", p_index);
                 for index1=1:repeat_time1
                    %fprintf("index1=%d\n", index1);
                    D=randn(p(p_index),r(r_default_index))*diag(rand(r(r_default_index),1)+0.5*ones(r(r_default_index),1));
                    D=D/norm(D);
                    X=random_ini_X(r(r_default_index),n(n_default_index),theta);
                    E=generate_heter_noise(noise_cata{cata_index},0.1,n(n_default_index),p(p_index));
                    Y=D*X+sqrt(noise_signal_ratio(eps_default_index)/p(p_index))*E;
                    [U,S,V]=svd(Y);
                    s=diag(S);
                    s_sqr=s.^2;
                    Y_proj=sqrt(n(n_default_index))*V(:,1:r(r_default_index))';
                    noise_variance=sum(s_sqr(r(1)+1:length(s)))/(length(s)-r(r_default_index));
                    s_modi=s_sqr(1:r(r_default_index))-noise_variance*ones([r(r_default_index),1]);
                    s_modi_inv_sqrt=ones([r(r_default_index),1])./s_modi.^(1/2);
                    Y_new=diag(s_modi_inv_sqrt)*diag(s(1:r(r_default_index)))*Y_proj;
                    for index2=1:repeat_time2
                        is_modifiedpgd=false;
                        if_ini_spliting=false;
                        [D_l4]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum=error_L4_sum+error_D_l4;

                        %before gradient modi
                        is_modifiedpgd=false;
                        if_ini_spliting=false;
                        [D_l4]=GD_L4(Y_new,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum_before=error_L4_sum_before+error_D_l4;

                        is_modifiedpgd=true;
                        if_ini_spliting=false;
                        [D_l4]=GD_L4(Y_new,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s_modi.^(1/2))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum_after=error_L4_sum_after+error_D_l4;

                        is_modifiedpgd=true;
                        if_ini_spliting=true;
                        [D_l4]=GD_L4(Y_new,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s_modi.^(1/2))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum_final=error_L4_sum_final+error_D_l4;
                    end
                 end
                 error_p(1,cata_index,p_index)=error_L4_sum/(repeat_time1*repeat_time2);
                 error_p(2,cata_index,p_index)=error_L4_sum_before/(repeat_time1*repeat_time2);
                 error_p(3,cata_index,p_index)=error_L4_sum_after/(repeat_time1*repeat_time2);
                 error_p(4,cata_index,p_index)=error_L4_sum_final/(repeat_time1*repeat_time2);
    end
end
%}
for cata_index=1:length(noise_cata)
    for eps_index=1:length(noise_signal_ratio)
            error_L4_sum=0;
            error_L4_sum_before=0;
            error_L4_sum_after=0;
            error_L4_sum_final=0;
            fprintf("eps index=%d\n", eps_index);
                 for index1=1:repeat_time1
                    %fprintf("index1=%d\n", index1);
                    D=randn(p(p_default_index),r(r_default_index))*diag(rand(r(r_default_index),1)+0.5*ones(r(r_default_index),1));
                    D=D/norm(D);
                    X=random_ini_X(r(r_default_index),n(n_default_index),theta);
                    E=generate_heter_noise(noise_cata{cata_index},0.1,n(n_default_index),p(p_default_index));
                    Y=D*X+sqrt(noise_signal_ratio(eps_index)/p(p_default_index))*E;
                    [U,S,V]=svd(Y);
                    s=diag(S);
                    s_sqr=s.^2;
                    Y_proj=sqrt(n(n_default_index))*V(:,1:r(r_default_index))';
                    noise_variance=sum(s_sqr(r(1)+1:length(s)))/(length(s)-r(r_default_index));
                    s_modi=s_sqr(1:r(r_default_index))-noise_variance*ones([r(r_default_index),1]);
                    s_modi_inv_sqrt=ones([r(r_default_index),1])./s_modi.^(1/2);
                    Y_new=diag(s_modi_inv_sqrt)*diag(s(1:r(r_default_index)))*Y_proj;
                    for index2=1:repeat_time2
                        is_modifiedpgd=false;
                        if_ini_spliting=false;
                        [D_l4]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum=error_L4_sum+error_D_l4;

                        %before gradient modi
                        is_modifiedpgd=false;
                        if_ini_spliting=false;
                        [D_l4]=GD_L4(Y_new,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum_before=error_L4_sum_before+error_D_l4;

                        is_modifiedpgd=true;
                        if_ini_spliting=false;
                        [D_l4]=GD_L4(Y_new,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s_modi.^(1/2))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum_after=error_L4_sum_after+error_D_l4;

                        is_modifiedpgd=true;
                        if_ini_spliting=true;
                        [D_l4]=GD_L4(Y_new,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                         D_l4proj=proj_orthogonal_group(D_l4);
                        D_rec_L4=U(:,1:r(r_default_index))*diag(s_modi.^(1/2))*D_l4proj;
                        D_rec_L4=D_rec_L4/norm(D_rec_L4);
                        [error_D_l4,~,~]=error3(D_rec_L4,D);
                        error_L4_sum_final=error_L4_sum_final+error_D_l4;
                    end
                 end
                 error_eps(1,cata_index,eps_index)=error_L4_sum/(repeat_time1*repeat_time2);
                 error_eps(2,cata_index,eps_index)=error_L4_sum_before/(repeat_time1*repeat_time2);
                 error_eps(3,cata_index,eps_index)=error_L4_sum_after/(repeat_time1*repeat_time2);
                 error_eps(4,cata_index,eps_index)=error_L4_sum_final/(repeat_time1*repeat_time2);
    end
end

save('Grad_improvement','error_p','error_eps')