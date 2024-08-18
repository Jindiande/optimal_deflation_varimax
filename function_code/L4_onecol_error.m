
% This function aims to solve maximise ||q'*Y||_4^{4} q\in O(p,1)
function [error_q]=L4_onecol_error(Y,D,maxstep)% Y should be processed 
[p,n]=size(Y);
q_init=randn(p,1);
q=L4_onecolumn(q_init,Y,maxstep);
error_q=min(ones(size(D'*q))-abs(D'*q));% D should be normlised

end