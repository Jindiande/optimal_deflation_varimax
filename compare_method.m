clear all; clc;
addpath('./function_code');
p=[10,20,30,50,70,100,300,700];
r=[3,5,10,15,20];
% r=[10:5:50]
theta=0.1;
noise_signal_ratio=[0.1:0.1:0.7];
maxstep=60;
alpha_log=[-1:0.1:0];
TOL = 1e-5;
tau=0.1;
MAX_ITER=10^5;
%noise_cata={'version_1','version_2','version_3'};
noise_cata={'version_1'};
method_cata={'PCA-dVarimax','PCA','Sparse-PCA','Varimax','Fast-ICA','Fast-ICA-AuddyandYuan'};
%n=1.2*10^(4);
n=[50,100,300,500,700,900];
prob_type=4;
if_nullspace=true;
if_pgd=false;
if_pertub_onec=false;
if_pertub_full=false;
if_inisvd=false;
if_earlystop=false;
if_checkrepeat=false;
if_tensor_ini=false;
if_ini_spliting=false;
if_mrs=false;
% if_centering=false;
is_modifiedpgd=false;
is_inidenoise=false;
perturb_rate=0;
repeat_time1=50;
repeat_time2=1;
repeat_time_ini=100;

p_default_index=7;
r_default_index=2;
eps_default_index=1;
n_default_index=6;

error_n=zeros(length(method_cata),length(noise_cata),length(n));
error_p=zeros(length(method_cata),length(noise_cata),length(p));
error_r=zeros(length(method_cata),length(noise_cata),length(r));
error_eps=zeros(length(method_cata),length(noise_cata),length(noise_signal_ratio));


%varing n
fprintf("varing n\n");
for cata_index=1:length(noise_cata)
for n_index=1:length(n)
        error_PCA_defla_sum=0;
        error_pca_sum=0;
        error_spca_sum=0;
        error_varimax_sum=0;
        error_ica_sum=0;
        error_Ay_sum=0;
             for index1=1:repeat_time1
                D=randn(p(p_default_index),r(r_default_index))*diag(rand(r(r_default_index),1)+0.5*ones(r(r_default_index),1));
                D=D/norm(D);
                X=random_ini_X(r(r_default_index),n(n_index),theta);
                E=generate_heter_noise(noise_cata{cata_index},0.1,n(n_index),p(p_default_index));
                Y=D*X+sqrt(noise_signal_ratio(eps_default_index)/p(p_default_index))*E;
                [U,S,V]=svd(Y);
                s=diag(S);
                Y_proj=V(:,1:r(r_default_index))';
                for index2=1:repeat_time2
                    if_checkrepeat=false;
                    repeat_time_ini=r(r_default_index)^2;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=true;
                    if_nullspace=false;
                    prob_type=4;
                    [D_l4]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_PCA_defla,~,~]=error3(D_rec_L4,D);
                     error_PCA_defla_sum=error_PCA_defla_sum+error_PCA_defla;%pca_deflation

                    D_rec_pca=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*eye(r(r_default_index));
                     D_rec_pca=D_rec_pca/norm(D_rec_pca);
                    [error_D_pca,~,~]=error3(D_rec_pca,D);
                    error_pca_sum=error_pca_sum+error_D_pca;

                    [~,D_res_adm,X]=ADMM(Y,D,MAX_ITER,TOL,tau);
                    D_res_adm=D_res_adm*diag(diag(X*X').^(1/2))/max(diag(X*X').^(1/2));
                    [error_D_spca,~,~]=error3(D_res_adm,D);
                    error_spca_sum=error_spca_sum+error_D_spca;

                    [~,D_res_varimax]=rotatefactors(Y_proj','Method','orthomax','Normalize','off','Maxit',1000);
                     D_rec_varimax=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_res_varimax;
                     D_rec_varimax=D_rec_varimax/norm(D_rec_varimax);
                    [error_D_varimax,~,~]=error3(D_rec_varimax,D);
                    error_varimax_sum=error_varimax_sum+error_D_varimax;

                    [~, W, ~, ~]=fastICA(Y,r(r_default_index));
                    D_rec_ica=W'*diag(s(1:r(r_default_index)));
                    D_rec_ica=D_rec_ica/norm(D_rec_ica);
                    [error_D_ica,~,~]=error3(D_rec_ica,D);
                    error_ica_sum=error_ica_sum+error_D_ica;  

                    if_checkrepeat=false;
                    %repeat_time_ini=r(r_default_index)^2;
                    repeat_time_ini=1;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=false;
                    if_nullspace=true;
                    prob_type=3;
                    [Zw, ~] = whitenRows(Y);
                    [D_l4]=GD_L4(Zw,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=D_l4proj*diag(s(1:r(r_default_index)));
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_ay,~,~]=error3(D_rec_L4,D);
                     error_Ay_sum=error_Ay_sum+error_ay;
                end
             end
             error_n(1,cata_index,n_index)=error_PCA_defla_sum/(repeat_time1*repeat_time2);
             error_n(2,cata_index,n_index)=error_pca_sum/(repeat_time1*repeat_time2);
             error_n(3,cata_index,n_index)=error_spca_sum/(repeat_time1*repeat_time2);
             error_n(4,cata_index,n_index)=error_varimax_sum/(repeat_time1*repeat_time2);
             error_n(5,cata_index,n_index)=error_ica_sum/(repeat_time1*repeat_time2);
             error_n(6,cata_index,n_index)=error_Ay_sum/(repeat_time1*repeat_time2);


end
end
%}

%varing p
fprintf("varing p\n");
for cata_index=1:length(noise_cata)
for p_index=1:length(p)
        error_PCA_defla_sum=0;
        error_pca_sum=0;
        error_spca_sum=0;
        error_varimax_sum=0;
        error_ica_sum=0;
        error_Ay_sum=0;
        %D=randn(p(p_index),r(1))*diag(rand(r(1),1)+0.5*ones(r(1),1));
        %D=D/norm(D);
             for index1=1:repeat_time1
                D=randn(p(p_index),r(r_default_index))*diag(rand(r(r_default_index),1)+0.5*ones(r(r_default_index),1));
                D=D/norm(D);
                X=random_ini_X(r(r_default_index),n(n_default_index),theta);
                E=generate_heter_noise(noise_cata{cata_index},0.1,n(n_default_index),p(p_index));
                Y=D*X+sqrt(noise_signal_ratio(eps_default_index)/p(p_index))*E;
                [U,S,V]=svd(Y);
                s=diag(S);
                Y_proj=V(:,1:r(r_default_index))';
                for index2=1:repeat_time2
                    if_checkrepeat=false;
                    repeat_time_ini=r(r_default_index)^2;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=true;
                    if_nullspace=false;
                    prob_type=4;
                    [D_l4]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_PCA_defla,~,~]=error3(D_rec_L4,D);
                     error_PCA_defla_sum=error_PCA_defla_sum+error_PCA_defla;%pca_deflation

                    D_rec_pca=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*eye(r(r_default_index));
                     D_rec_pca=D_rec_pca/norm(D_rec_pca);
                    [error_D_pca,~,~]=error3(D_rec_pca,D);
                    error_pca_sum=error_pca_sum+error_D_pca;

                    [~,D_res_adm,X]=ADMM(Y,D,MAX_ITER,TOL,tau);
                    D_res_adm=D_res_adm*diag(diag(X*X').^(1/2))/max(diag(X*X').^(1/2));
                    [error_D_spca,~,~]=error3(D_res_adm,D);
                    error_spca_sum=error_spca_sum+error_D_spca;

                    [~,D_res_varimax]=rotatefactors(Y_proj','Method','orthomax','Normalize','off','Maxit',1000);
                     D_rec_varimax=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_res_varimax;
                     D_rec_varimax=D_rec_varimax/norm(D_rec_varimax);
                    [error_D_varimax,~,~]=error3(D_rec_varimax,D);
                    error_varimax_sum=error_varimax_sum+error_D_varimax;

                    [~, W, ~, ~]=fastICA(Y,r(r_default_index));
                    D_rec_ica=W'*diag(s(1:r(r_default_index)));
                    D_rec_ica=D_rec_ica/norm(D_rec_ica);
                    [error_D_ica,~,~]=error3(D_rec_ica,D);
                    error_ica_sum=error_ica_sum+error_D_ica;  

                    if_checkrepeat=false;
                    %repeat_time_ini=r(r_default_index)^2;
                    repeat_time_ini=1;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=false;
                    if_nullspace=true;
                    prob_type=3;
                    [Zw, ~] = whitenRows(Y);
                    [D_l4]=GD_L4(Zw,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=D_l4proj*diag(s(1:r(r_default_index)));
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_ay,~,~]=error3(D_rec_L4,D);
                     error_Ay_sum=error_Ay_sum+error_ay;
                end
             end
             error_p(1,cata_index,p_index)=error_PCA_defla_sum/(repeat_time1*repeat_time2);
             error_p(2,cata_index,p_index)=error_pca_sum/(repeat_time1*repeat_time2);
             error_p(3,cata_index,p_index)=error_spca_sum/(repeat_time1*repeat_time2);
             error_p(4,cata_index,p_index)=error_varimax_sum/(repeat_time1*repeat_time2);
             error_p(5,cata_index,p_index)=error_ica_sum/(repeat_time1*repeat_time2);
             error_p(6,cata_index,p_index)=error_Ay_sum/(repeat_time1*repeat_time2);
end
end
%}
%varing r
%{
fprintf("varing r\n");
for cata_index=1:length(noise_cata)
for r_index=1:length(r)
        error_PCA_defla_sum=0;
        error_pca_sum=0;
        error_spca_sum=0;
        error_varimax_sum=0;
        error_ica_sum=0;
        error_Ay_sum=0;
        fprintf("r_index=%d\n",r_index);
              for index1=1:repeat_time1
                D=randn(p(p_default_index),r(r_index))*diag(rand(r(r_index),1)+0.5*ones(r(r_index),1));
                D=D/norm(D);
                X=random_ini_X(r(r_index),n(n_default_index),theta);
                E=generate_heter_noise(noise_cata{cata_index},0.1,n(n_default_index),p(p_default_index));
                Y=D*X+sqrt(noise_signal_ratio(eps_default_index)/p(p_default_index))*E;
                [U,S,V]=svd(Y);
                s=diag(S);
                Y_proj=V(:,1:r(r_index))';
                for index2=1:repeat_time2
                    if_checkrepeat=false;
                    repeat_time_ini=r(r_index)^2;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=true;
                    if_nullspace=false;
                    prob_type=4;
                    [D_l4]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=U(:,1:r(r_index))*diag(s(1:r(r_index)))*D_l4proj;
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_PCA_defla,~,~]=error3(D_rec_L4,D);
                     error_PCA_defla_sum=error_PCA_defla_sum+error_PCA_defla;%pca_deflation

                    D_rec_pca=U(:,1:r(r_index))*diag(s(1:r(r_index)))*eye(r(r_index));
                     D_rec_pca=D_rec_pca/norm(D_rec_pca);
                    [error_D_pca,~,~]=error3(D_rec_pca,D);
                    error_pca_sum=error_pca_sum+error_D_pca;
                    [~,D_res_adm,X]=ADMM(Y,D,MAX_ITER,TOL,tau);
                    D_res_adm=D_res_adm*diag(diag(X*X').^(1/2))/max(diag(X*X').^(1/2));
                    [error_D_spca,~,~]=error3(D_res_adm,D);
                    error_spca_sum=error_spca_sum+error_D_spca;

                    [~,D_res_varimax]=rotatefactors(Y_proj','Method','orthomax','Normalize','off','Maxit',1000);
                     D_rec_varimax=U(:,1:r(r_index))*diag(s(1:r(r_index)))*D_res_varimax;
                     D_rec_varimax=D_rec_varimax/norm(D_rec_varimax);
                    [error_D_varimax,~,~]=error3(D_rec_varimax,D);
                    error_varimax_sum=error_varimax_sum+error_D_varimax;

                    [~, W, ~, ~]=fastICA(Y,r(r_index));
                    D_rec_ica=W'*diag(s(1:r(r_index)));
                    D_rec_ica=D_rec_ica/norm(D_rec_ica);
                    [error_D_ica,~,~]=error3(D_rec_ica,D);
                    error_ica_sum=error_ica_sum+error_D_ica;  

                    if_checkrepeat=false;
                    %repeat_time_ini=r(r_default_index)^2;
                    repeat_time_ini=1;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=false;
                    if_nullspace=true;
                    prob_type=3;
                    [Zw, ~] = whitenRows(Y);
                    [D_l4]=GD_L4(Zw,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=D_l4proj*diag(s(1:r(r_index)));
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_ay,~,~]=error3(D_rec_L4,D);
                     error_Ay_sum=error_Ay_sum+error_ay;
                end
             end
             error_r(1,cata_index,r_index)=error_PCA_defla_sum/(repeat_time1*repeat_time2);
             error_r(2,cata_index,r_index)=error_pca_sum/(repeat_time1*repeat_time2);
             error_r(3,cata_index,r_index)=error_spca_sum/(repeat_time1*repeat_time2);
             error_r(4,cata_index,r_index)=error_varimax_sum/(repeat_time1*repeat_time2);
             error_r(5,cata_index,r_index)=error_ica_sum/(repeat_time1*repeat_time2);
             error_r(6,cata_index,r_index)=error_Ay_sum/(repeat_time1*repeat_time2);
end
end
%}
%{
%varing noise level
fprintf("varing noise\n");
for cata_index=1:length(noise_cata)
for noise_index=1:length(noise_signal_ratio)
        error_PCA_defla_sum=0;
        error_pca_sum=0;
        error_spca_sum=0;
        error_varimax_sum=0;
        error_ica_sum=0;
        error_Ay_sum=0;
              for index1=1:repeat_time1
                D=randn(p(p_default_index),r(r_default_index))*diag(rand(r(r_default_index),1)+0.5*ones(r(r_default_index),1));
                D=D/norm(D);
                X=random_ini_X(r(r_default_index),n(n_default_index),theta);
                E=generate_heter_noise(noise_cata{cata_index},0.1,n(n_default_index),p(p_default_index));
                Y=D*X+sqrt(noise_signal_ratio(noise_index)/p(p_default_index))*E;
                [U,S,V]=svd(Y);
                s=diag(S);
                Y_proj=V(:,1:r(r_default_index))';
                for index2=1:repeat_time2
                    if_checkrepeat=false;
                    repeat_time_ini=r(r_default_index)^2;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=true;
                    if_nullspace=false;
                    prob_type=4;
                    [D_l4]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_l4proj;
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_PCA_defla,~,~]=error3(D_rec_L4,D);
                     error_PCA_defla_sum=error_PCA_defla_sum+error_PCA_defla;%pca_deflation

                    D_rec_pca=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*eye(r(r_default_index));
                     D_rec_pca=D_rec_pca/norm(D_rec_pca);
                    [error_D_pca,~,~]=error3(D_rec_pca,D);
                    error_pca_sum=error_pca_sum+error_D_pca;
                    [~,D_res_adm,X]=ADMM(Y,D,MAX_ITER,TOL,tau);
                    D_res_adm=D_res_adm*diag(diag(X*X').^(1/2))/max(diag(X*X').^(1/2));
                    [error_D_spca,~,~]=error3(D_res_adm,D);
                    error_spca_sum=error_spca_sum+error_D_spca;

                    [~,D_res_varimax]=rotatefactors(Y_proj','Method','orthomax','Normalize','off','Maxit',1000);
                     D_rec_varimax=U(:,1:r(r_default_index))*diag(s(1:r(r_default_index)))*D_res_varimax;
                     D_rec_varimax=D_rec_varimax/norm(D_rec_varimax);
                    [error_D_varimax,~,~]=error3(D_rec_varimax,D);
                    error_varimax_sum=error_varimax_sum+error_D_varimax;

                    [~, W, ~, ~]=fastICA(Y,r(r_default_index));
                    D_rec_ica=W'*diag(s(1:r(r_default_index)));
                    D_rec_ica=D_rec_ica/norm(D_rec_ica);
                    [error_D_ica,~,~]=error3(D_rec_ica,D);
                    error_ica_sum=error_ica_sum+error_D_ica;  

                    if_checkrepeat=false;
                    %repeat_time_ini=r(r_default_index)^2;
                    repeat_time_ini=1;
                    if_ini_spliting=false;
                    if_tensor_ini=false;
                    if_mrs=false;
                    if_nullspace=true;
                    prob_type=3;
                    [Zw, ~] = whitenRows(Y);
                    [D_l4]=GD_L4(Zw,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
                     D_l4proj=proj_orthogonal_group(D_l4);
                     D_rec_L4=D_l4proj*diag(s(1:r(r_default_index)));
                     D_rec_L4=D_rec_L4/norm(D_rec_L4);
                     [error_ay,~,~]=error3(D_rec_L4,D);
                     error_Ay_sum=error_Ay_sum+error_ay;
                end
             end
             error_eps(1,cata_index,noise_index)=error_PCA_defla_sum/(repeat_time1*repeat_time2);
             error_eps(2,cata_index,noise_index)=error_pca_sum/(repeat_time1*repeat_time2);
             error_eps(3,cata_index,noise_index)=error_spca_sum/(repeat_time1*repeat_time2);
             error_eps(4,cata_index,noise_index)=error_varimax_sum/(repeat_time1*repeat_time2);
             error_eps(5,cata_index,noise_index)=error_ica_sum/(repeat_time1*repeat_time2);
             error_eps(6,cata_index,noise_index)=error_Ay_sum/(repeat_time1*repeat_time2);
end
end
%}
load('compare_22-Jul-2024.mat','error_eps','error_r');
str1=strsplit(datestr(datetime));
str1=strcat('compare_',str1{1});
save(str1,'error_n','error_p','error_r','error_eps');