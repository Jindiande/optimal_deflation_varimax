clear all; clc;
prob_type=4;
if_nullspace=true;
if_pgd=false;
if_pertub_onec=false;
if_pertub_full=false;
if_inisvd=false;
if_earlystop=false;
if_checkrepeat=false;
if_centering=true;
is_modifiedpgd=false;
is_newdata=true;
is_minst=true;
addpath('../test_slate');
addpath('../function_code');
addpath('../data_image');
Dir = dir(fullfile('../test_slate', '*.pgm')); 
patch_size=28;
p=patch_size^2;
r=49;
image_num=8000;
if (if_pgd || if_earlystop)
    maxstep=5*10^3;
else
    maxstep=60;
end

fig = figure('Visible', 'on');
if is_minst
    load('../MINST/Minst_image.mat');
elseif is_newdata
    load('../data_image/IMAGES.mat');
end
%reconstruction

Y=[];
for i = 1:image_num
        %tlo =tiledlayout(round(sqrt(r)),round(sqrt(r)));
        %tlo1 =tiledlayout(round(sqrt(r)),round(sqrt(r)));
        %disp('Loading the image!')
        
        if is_minst
            I=squeeze(extractdata(Minst(:,:,1,i)));
            size(I);
            %I=double(IMAGES(:,:,i));
        elseif is_newdata
            %I=double(Minst(:,:,1,i));
            I=double(IMAGES(:,:,i));
        else
            img = imread(Dir(i).name);
            if ndims(img) > 2, 
                I = double(rgb2gray(imread(img)));
            else
                I = double(img);
            end
        end
        if(patch_size<28)
            Y_i = image_to_patches(I, patch_size);
        else
            Y_i=reshape(I,[patch_size^2,1]);
        end
        if (if_centering)
            %[Y,mu_r,mu_c,mu]=centering(Y);
            Y_i=Y_i-repmat(mean(Y_i,1),size(Y_i,1),1);
            %Y_i=Y_i-repmat(mean(Y_i,2),1,size(Y_i,2));
        end
        Y=[Y,Y_i];
        rng(i, 'twister');  % to ensure reproducibility, we specify both the seed and the alg 
        %}
        %X_rec=pinv(A)*Y;
        %{
        [D_l4]=GD_L4(Y,eye(r(r_index)),maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,0,if_inisvd,if_earlystop,if_checkrepeat); 
        D_l4=proj_orthogonal_group(D_l4);
        X_rec=D_l4'*Y;
        Y_rec=D_l4*X_rec;
        %}
        %Y_rec=A*X_rec;
        % PCA
        %{
        [U,S,V]=svd(Y);
        Y_pca=U(:,1:r(r_index))*S(1:r(r_index),1:r(r_index))*(V(:,1:r(r_index)))';

        if (if_centering)
            [Y_rec]=decentering(Y_rec,mu_r,mu_c,mu);
            [Y_pca]=decentering(Y_pca,mu_r,mu_c,mu);
        end
        %}
        %{
        figure(1);
        set(gcf,'PaperPositionMode','auto'); 
        print(gcf, '-dpdf', fullfile('results', [num2str(i), '_img.pdf'])); 
        system(['pdfcrop ', fullfile('results', [num2str(i), '_img.pdf'])]); 

        figure(2);
        set(gcf,'PaperPositionMode','auto'); 
        print(gcf, '-dpdf', fullfile('results', [num2str(i), '_dict.pdf'])); 
        system(['pdfcrop ', fullfile('results', [num2str(i), '_dict.pdf'])]); 

        figure(3);
        set(gcf,'PaperPositionMode','auto'); 
        print(gcf, '-dpdf', fullfile('results', [num2str(i), '_obj.pdf'])); 
        system(['pdfcrop ', fullfile('results', [num2str(i), '_obj.pdf'])]); 
    %}
        end
%}

%
tlo =tiledlayout(round(sqrt(r)),round(sqrt(r)));
% our method
%Y=Y(:, randperm(size(Y,2)));
[Y_proj,U,s]=pre_condi(Y,r);
[D_l4]=GD_L4(Y_proj,eye(r),maxstep,prob_type,if_nullspace,if_pgd,if_pertub_onec,if_pertub_full,0,if_inisvd,if_earlystop,if_checkrepeat,is_modifiedpgd,s.^2); 
X_rec=D_l4'*Y_proj;
D_l4=proj_orthogonal_group(D_l4);
D_rec_L4=U(:,1:r)*diag(s(1:r))*D_l4;

for j=1:size(D_rec_L4,2)
    nexttile(tlo)
    I_rec=patches_to_image(D_rec_L4(:,j));
    imagesc(I_rec);
    colormap gray; 
    axis off;
end
%}
figure;
tlo1 =tiledlayout(round(sqrt(r)),round(sqrt(r)));

for j=1:size(D_rec_L4,2)
    nexttile(tlo1)
    I_rec=patches_to_image(U(:,j));
    imagesc(I_rec);
    colormap gray; 
    axis off;
end
%}


%tiledlayout(r-1,2);

for i =1:5
    figure()
    plot(X_rec(2*i-1,:),X_rec(2*i,:),'linestyle','none','marker','.');
    xline(0);
    yline(0);
    title("PCA-dVarimax: "+'No.'+num2str(i)+"pair")
    %saveas(gcf,"RealImageDictionary_result/"+"Our_Method_"+'No'+num2str(i)+"pair"+"image",'epsc')
    hold off;
    figure()
    x=Y_proj(2*i-1,:);
    x=x-mean(x);
    y=Y_proj(2*i,:);
    y=y-mean(y);
    plot(x,y,'linestyle','none','marker','.');
    xline(0);
    yline(0);
    title("PCA: "+'No.'+num2str(i)+"pair")
    %saveas(gcf,"RealImageDictionary_result/"+"PCA"+'_No'+num2str(i)+"pair"+"image",'epsc')
    hold off;
end
%}
%h = histogram(X_rec)
%saveas(gcf,"real_data_image/"+"result",'epsc')