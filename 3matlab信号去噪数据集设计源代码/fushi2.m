function d=fushi2(dat,g,op)
%op��0--g��ԭ���ڿ�ʼ�㣬1--g��ԭ�������ĵ㣨Ҫ��g����Ϊ��������2--g��ԭ����ĩ�㣻Ŀǰֻ֧�ִ�3��
 
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
            if tmp < dtmp(i) %�����С��ԭ����ֵ��������Сֵ��
                dtmp(i) = tmp;
            end
        end
	end
end
% toc
 d = dtmp;
