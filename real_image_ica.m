
clear all; clc;
prob_type=3;
if_nullspace=true;
if_pgd=false;
if_pertub_onec=false;
if_pertub_full=false;
if_inisvd=false;
if_earlystop=false;
if_checkrepeat=false;
if_centering=true;
is_modifiedpgd=false;
is_additionalnoise=true;
is_inidenoise=false;
if_ini_spliting=false;
perturb_rate=0;
addpath('../test_slate');
addpath('../function_code');
Dir = dir(fullfile('../test_slate', '*.pgm')); 
index_list={'First','Second','Third','Fourth','Fifth','Sixth','Seventh','Eighth','Ninth'};
%patch_size=8;
%p=patch_size^2;
epsq=100;
r=5;
num=2;
num_print=5;
p=100;
if (if_pgd || if_earlystop)
    maxstep=5*10^3;
else
    maxstep=70;
end
fig = figure('Visible', 'on');
%reconstruction
Y=[];

for i = 1:num
        %tlo =tiledlayout(round(sqrt(r)),round(sqrt(r)));
        %tlo1 =tiledlayout(round(sqrt(r)),round(sqrt(r)));
        disp('Loading the image!');
        img = imread(Dir(i+2).name);

        if ndims(img) > 2, 
            I = double(rgb2gray(imread(img)));
        else
            I = double(img);
        end
        I=imresize(I,[round(size(I,1))/3 round(size(I,2)/3)]);%downsample
        Y_i = image_to_patches(I, size(I,1));%reshape
        size(Y_i)
        Y=[Y;Y_i'];
end
tlo =tiledlayout(2,r);

% mix noise
for i =1:(r-num)
    Y=[Y;epsq*randn(1,size(Y,2))];
end
size(Y)
for j=1:size(Y,1)
    nexttile(tlo)
    I_rec=patches_to_image(Y(j,:)');
    imagesc(I_rec);
    colormap gray; 
    axis off;
end
%generate Y
%A=proj_orthogonal_group(randn(size(Y,1),size(Y,1)));
A=randn(p,size(Y,1));
%A=col_wise_norm(A);
A=A/norm(A);
Y=A*Y;
%centering
if (if_centering)
    %[Y,mu_r,mu_c,mu]=centering(Y);
    Y=Y-repmat(mean(Y,2),1,size(Y,2));
end
if(is_additionalnoise)
   Y=Y+0.6*randn(size(Y,1),size(Y,2));
end

for j=1:r
    nexttile(tlo)
    I_rec=patches_to_image(Y(j,:)');
    imagesc(I_rec);
    colormap gray; 
    axis off;
end
mean(Y,2);
%FAST-ICA
[Y_ica, W, T, mu]=fastICA(Y,num_print);

[Y_proj,U,s]=pre_condi(Y,num_print);
s_sqr=s.^2;

noise_variance=sum(s_sqr(r+1:length(s)))/(length(s)-r);
s_modi=s_sqr(1:r)-noise_variance*ones([r,1]);
s_modi_inv_sqrt=ones([r,1])./s_modi.^(1/2);
Y_new=diag(s_modi_inv_sqrt)*diag(s(1:r))*Y_proj;


%PCA-dVarimax
repeat_time_ini=1;
if_tensor_ini_spliting=false;
if_tensor_ini=false;
if_mrs=false;
if_nullspace=true;
is_inidenoise=false;
prob_type=3;
[D_l4]=GD_L4(Y_proj,eye(num_print),maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
D_l4=proj_orthogonal_group(D_l4);
Y_rec=diag(s(1:num_print))*D_l4'*Y_proj;
%Fast-ICA-AuddyYuan
repeat_time_ini=25;
if_tensor_ini_spliting=false;
if_tensor_ini=false;
if_mrs=false;%if if_tensor_ini is true, this para is abandoned.
if_nullspace=true;%if if_tensor_ini is true, this para is abandoned.
prob_type=1;%if if_tensor_ini is true, this must be 1
is_inidenoise=false;
[Y_ica_am,~] = whitenRows(Y);
[D_l4]=GD_L4(Y_ica_am,eye(num_print),maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
D_l4=proj_orthogonal_group(D_l4);
Y_rec_ica_ay=diag(s(1:num_print))*D_l4'*Y_ica_am;

% PCA-dVarimax-1
% repeat_time_ini=r^2;
% if_nullspace=true;
% if_tensor_ini_spliting=false;
% if_tensor_ini=false;
% if_mrs=false;
% prob_type=4;
% [D_l4_mri]=GD_L4(Y_proj,D,maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,if_tensor_ini,if_tensor_ini_spliting,if_mrs,s.^2,repeat_time_ini); 
% D_l4_mri=proj_orthogonal_group(D_l4);



%PCA-dVarimax denoise
is_modifiedpgd=true;
prob_type=3;
[D_l4_denoise]=GD_L4(Y_new,eye(num_print),maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,perturb_rate,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,is_inidenoise,if_tensor_ini,if_ini_spliting,if_mrs,s.^2,repeat_time_ini);
D_l4_denoise=proj_orthogonal_group(D_l4_denoise);
Y_rec_denoise=(diag(s_sqr(1:num_print))-noise_variance*eye(num_print))^(1/2)*D_l4_denoise'*Y_new;
%Y_rec=s(1)*D_l4'*Y_proj;
B=rotatefactors(sqrt(size(Y,2))^(-1)*Y_proj','Method','orthomax','Normalize','off','Maxit',5000);
Y_varimax=s(1)*B';
Y_pca=s(1)*eye(num_print)*Y_proj;

% if (if_centering)
%     %[Y_rec]=decentering(Y_rec,mu_r,mu_c,mu);
% end
%}
figure;
tlo =tiledlayout(6,num_print);
%tlo.TileIndexing='columnmajor';
for j=1:num_print
     nexttile(tlo);
     if(j==1|| j==0)
         I_rec=patches_to_image(-Y_rec(j,:)');
     else      
         I_rec=patches_to_image(Y_rec(j,:)');
     end
     imagesc(I_rec);
     title("PCA-dVarimax: "+index_list{j}+' rotated PC');
     colormap gray; 
     axis off;
end

%PCA-dvarimax denoise
for j=1:num_print
     nexttile(tlo);
     if(j==1||j==num_print)
         I_rec=patches_to_image(-Y_rec_denoise(j,:)');
     else      
         I_rec=patches_to_image(Y_rec_denoise(j,:)');
     end
     imagesc(I_rec);
     title("PCA-dVarimax2: "+index_list{j}+' rotated PC');
     colormap gray; 
     axis off;
 end
%ica
for j=1:num_print
    nexttile(tlo);
    if(j==3)
        I_rec=patches_to_image(-Y_ica(j,:)');
    else      
        I_rec=patches_to_image(Y_ica(j,:)');
    end
    imagesc(I_rec);
    title("Fast-ICA: "+index_list{j}+' rotated PC');
    colormap gray; 
    axis off;
end
%ica auddy and yuan
for j=1:num_print
    nexttile(tlo);
    I_rec=patches_to_image(Y_rec_ica_ay(j,:)');
    imagesc(I_rec);
    title("Fast-ICA-Tensor: "+index_list{j}+' rotated PC');
    colormap gray; 
    axis off;
end
%pca

for j=1:num_print
    nexttile(tlo);
    if(j==0 || j==0)
        I_rec=patches_to_image(-Y_pca(j,:)');
    else      
        I_rec=patches_to_image(Y_pca(j,:)');
    end
    imagesc(I_rec);
    title("PCA: "+index_list{j}+' rotated PC');
    colormap gray; 
    axis off;
end
%save('real_image_ica','Y_rec','Y_ica','Y_pca');
%varimax

%B=rotatefactors(sqrt(size(Y,2))^(-1)*Y_proj','Method','orthomax','Maxit',1000);
%Y_varimax=s(1)*B';
%save('real_image_ica','Y_rec','Y_ica','Y_pca','Y_varimax','Y_recvar');
for j=1:num_print
    nexttile(tlo);
    I_rec=patches_to_image(Y_varimax(j,:)');
    imagesc(I_rec);
    title("Varimax: "+index_list{j}+' rotated PC');
    colormap gray; 
    axis off;
end

%Y_recvar
%{
for j=1:num_print
    nexttile(tlo);
    I_rec=patches_to_image(Y_recvar(:,j));
    imagesc(I_rec);
    title("varimax after PCA-dVarimax: "+'No.'+num2str(j)+"'s base");
    colormap gray; 
    axis off;
end
%}

%}
%mixing matrix