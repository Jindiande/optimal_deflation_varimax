function [q]=L4_onecolumn_vari_size(q_ini,Y,D,maxstep,error_bar,if_early_stop)
q=q_ini;
eta_ini=0.01;
p=size(q,1);
alpha=-0.9;
for step=1:maxstep
   dq=L4_Grad(q,Y); 
   %dq_p=dq;
   dq_p=(eye(p,p)-q*q')*dq;
   eta = step^(alpha)*eta_ini;
   %eta=0.5;
   %t=norm(dq_p);
   %q=cos(t)*q+sin(t)*(dq_p/t);
   q=q+eta*dq_p;
   q=q/norm(q);
   if((if_early_stop)&&(min(ones(size(D'*q))-abs(D'*q))<error_bar))
       break;
   end
end