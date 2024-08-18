% to calculat M_0 in 'Aubby and Yuan, 2023'
% return shape [d,d,d,d]
function sum=calculate_M0(d)
% basis=pagemtimes(reshape(eye(r),[r,1,r]),reshape(eye(r),[1,r,r]));%shape r*r*r
sum=zeros([d,d,d,d]);
I=eye(d,d);
%the following part could be improved by parallel computing
for i = 1:d
    for j=1:d
        if i~=j
            for k=1:16 %2^4
                strlist=dec2bin(k);%string of binary number
                if(length(strlist)<4)
                    for l =1:(4-length(strlist))
                        strlist=append('0',strlist);
                    end
                end
                numlist=zeros(4,1);
                for l=1:length(strlist)%
                    if(strlist(l)=='1')
                        numlist(l,1)=i;
                    else
                        numlist(l,1)=j;
                    end
                end
                sum=sum+tensorprod(I(:,numlist(1,1))*I(:,numlist(2,1))',I(:,numlist(3,1))*I(:,numlist(4,1))');
            end
        else
            numlist=i*ones(4,1);
            sum=sum+tensorprod(I(:,numlist(1,1))*I(:,numlist(2,1))',I(:,numlist(3,1))*I(:,numlist(4,1))');
        end
    end
%sum=reshape(sum,[d^2,d^2]);
end