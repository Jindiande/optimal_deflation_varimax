function dA=L4_Grad(A,Y)
M1=Y'*A;
M2=M1.*M1.*M1;
dA=4*Y*M2;
end