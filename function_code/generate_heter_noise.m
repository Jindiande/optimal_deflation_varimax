function N=generate_heter_noise(version,alpha,n,p)% noise dimension [p,n] 
   if version=='version_2'%
        vari=rand(p,1);
        vari=vari.^(10^(alpha));
        vari=p*(vari/sum(vari));
        N=randn(p,n);
        N=repmat(vari,1,n).*N;
   elseif version=='version_1'%iid gaussian
        N=randn(p,n);  
   else
       mu=zeros(p,1);
       sig_arr_ini=transpose([0:1:p-1]);
       sig_arr=sig_arr_ini;
       sigma=[sig_arr];
       for i=2:p
           %size(sig_arr)
           sig_arr=cat(1,[sig_arr_ini(i)],sig_arr(1:p-1));
           sigma=[sigma,sig_arr];
       end
       N=transpose(mvnrnd(mu,(1/2).^sigma,n));
   end
end