function [Y_ori]=decentering(Y_cen,mu_r,mu_c,mu)
  Y_ori=Y_cen+mu_r*ones(1,size(Y_cen,2))+ones(size(Y_cen,1),1)*mu_c-ones(size(Y_cen,1),1)*mu*ones(1,size(Y_cen,2));
end