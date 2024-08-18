function q=L4_onecolumn(q_ini,Y,maxstep)
q=q_ini;
for step=1:maxstep
   dq=L4_Grad(q,Y); 
   q=dq/norm(dq);       
end
q=q;
end