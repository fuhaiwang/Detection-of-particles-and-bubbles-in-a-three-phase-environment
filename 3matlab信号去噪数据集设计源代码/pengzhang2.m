function d=pengzhang2(dat,g,op)
%op��0--g��ԭ���ڿ�ʼ�㣬1--g��ԭ�������ĵ㣨Ҫ��g����Ϊ��������2--g��ԭ����ĩ�㣻Ŀǰֻ֧�ִ�3��
len = length(dat);
len_g = length(g);
 
switch op
    case 1,
    org_piont = ceil(len_g/2);
    case 2,
      org_piont =  len_g;
  otherwise,
    org_piont = 1;
end
 
for i=1:len
    dtmp(i) = dat(i);
     for j=1:len_g
         if (i-org_piont+j) >= 1 && (i-org_piont+j) <=len
             tmp = dat(i-org_piont+j) + g(j);
           if tmp > dtmp(i)
               dtmp(i) = tmp;
           end
       end
    end
end
 
d = dtmp;