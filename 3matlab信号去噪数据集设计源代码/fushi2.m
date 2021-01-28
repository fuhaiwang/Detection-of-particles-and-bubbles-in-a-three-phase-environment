function d=fushi2(dat,g,op)
%op：0--g的原点在开始点，1--g的原点在中心点（要求g长度为奇数），2--g的原点在末点；目前只支持此3种
 
len = length(dat);
len_g = length(g);%100
 
switch op
    case 1,
    org_piont = ceil(len_g/2);%50
    case 2,
      org_piont =  len_g;
  otherwise,
    org_piont = 1;
 
end
% tic
for i=1:len
    dtmp(i) = dat(i);
	for j=1:len_g
        k=i-org_piont+j;%i-50+j
        if (k) >= 1 && (k) <=len
            tmp = dat(k) - g(j);
            if tmp < dtmp(i) %处理后小于原来的值，存下最小值。
                dtmp(i) = tmp;
            end
        end
	end
end
% toc
 d = dtmp;
