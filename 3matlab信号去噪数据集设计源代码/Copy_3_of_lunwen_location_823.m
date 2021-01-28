%四 %整体思路：小波阈值法去除高频——形态学去除工频——提取特征——保存。
%help wavelet %wavedemo
%wavemenu 小波工具箱 %https://ww2.mathworks.cn/help/deeplearning/signal-processing-using-deep-learning.html深度学习的matlab案例
close all  
clear all  %常用106,气泡70,80,混合126,148

for ii = 50:50 %批处理  70-100气泡  105-124颗粒  125-148混合   % 读取原始数据，按编号
    ii
    doc_Amplitude(ii,:) =['C:\Users\fuhai\Desktop\8-13\归一化\Amplitude813_',num2str(ii),'.xlsx']; %请改为自己的存储路径
    doc_pulse(ii,:)=['C:\Users\fuhai\Desktop\8-13\归一化\pulse10-8_',num2str(ii),'.xlsx',]; %请改为自己的存储路径
    %% 读入数据
    [num1]= xlsread(['C:\Users\fuhai\Desktop\8-13\Test813_',num2str(ii),'.xlsx'],2,'B3:B1048576');  %请改为自己的读取路径
    [num2]= xlsread(['C:\Users\fuhai\Desktop\8-13\Test813_',num2str(ii),'.xlsx'],1,'B3:B954428');   %请改为自己的读取路径
    num = [num1;num2];  %1048574 + 171010 = 1219584
    Alq=num';
    x=1000*Alq;%将幅度扩大1K倍，mv单位好看
    length1=size(Alq,2);%1219584
    tic
    %% 1.0原始数据绘制
    axis_daxiao=38;       %字体大小
    Fs = 200000;          % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = length1;          % Length of signal
    t = (0:L-1)*T;        % Time vector
    % %图一（a）原始信号的显示
    figure(2)
    subplot(1,2,1);
    plot(t,x,'LineWidth',2,'Color',[0,0,0])
    set(gca,'FontSize',axis_daxiao-10,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao) %时间 (s)
    ylabel('Voltage  (mv)','FontSize',axis_daxiao)

    Y = fft(x);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    subplot(1,2,2);
    plot(f,P1,'LineWidth',2,'Color',[0,0,0]) 
    
    axis([0.1 100000 0.0000005 10])  
    set(gca,'FontSize',axis_daxiao-8,'XScale','log') 
    set(gca,'FontSize',axis_daxiao-8,'YScale','log')
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('f (Hz)','FontSize',axis_daxiao)
    ylabel('FFT coefficients','FontSize',axis_daxiao)
    set(gcf,'unit','normalized','position',[0,0,1,0.6]);

    %% 2.0小波分解，W = wavenames('all')  bior4.4,db10,3,4都要好  不合适的：sym2，coif1
    % 小波细节，主要用于 高频去噪
    [c2,l2]=wavedec(x,4,'db10'); %重构第1~4层的细节信号
%     pd5=wrcoef('d',c2,l2,'db10',5);
    pd4=wrcoef('d',c2,l2,'db10',4);
    pd3=wrcoef('d',c2,l2,'db10',3);
    pd2=wrcoef('d',c2,l2,'db10',2);
    pd1=wrcoef('d',c2,l2,'db10',1);

    %% 2.1小波阈值法 求噪声方差
    %阈值法，去除高频干扰.
    %目前有多种阈值选取的方法。固定阈值估计sqtwolog，minimaxi、极值阈值估计、无偏似然估计rigrsure以及启发式估计等（N为信号长度）heursure  阈值函数的选择设计也有好多种
    %对小波细节系数进行小波软阈值滤波处理。确定阈值，有：统一阈值，无偏风险阈值，启发式阈值，MiniMax阈值。
    r1=median(abs(pd1))/0.6745*sqrt(2*log(length(pd1))); %var()求方差
    r2=median(abs(pd2))/0.6745*sqrt(2*log(length(pd2)));
    r3=median(abs(pd3))/0.6745*sqrt(2*log(length(pd3)));
    r4=median(abs(pd4))/0.6745*sqrt(2*log(length(pd4)));
    %% 2.2改进半阈值处理.系数越大，干扰越小
    for i=length(pd1):-1:1
        cd11(i)=0;
        if abs(pd1(i))>r1
            cd11(i)=sign(pd1(i))*sqrt(pd1(i)^2-r1^2);
        else
            cd11(i)=0;
        end
    end
    for i=length(pd2):-1:1
        cd21(i)=0;
        if abs(pd2(i))>r2
            cd21(i)=sign(pd2(i))*sqrt(pd2(i)^2-r2^2);
        else
            cd21(i)=0;
        end
    end
    for i=length(pd3):-1:1
        cd31(i)=0;
        if abs(pd3(i))>r3
            cd31(i)=sign(pd3(i))*sqrt(pd3(i)^2-r3^2);
        else
            cd31(i)=0;
        end
    end
    for i=length(pd4):-1:1
    	cd41(i)=0;
    	if abs(pd4(i))>r4
            cd41(i)=sign(pd4(i))*sqrt(pd4(i)^2-r4^2);
        else
            cd41(i)=0;
    	end
    end
    %% 2.3 阈值去噪显示信号结果
    p11=x-pd1-pd2-pd3+cd31-pd4+cd41; %+cd11+cd21;%+cd11;  高频去噪完毕
    p=x;
    %% 4.42小波阈值去噪 再加入频域信号绘图
    [fp11,Pp11]=spectrogram1(length1,p11); %写成一个函数了，直接给出FFT后的单边频谱图
    [fp1,Pp1]=spectrogram1(length1,p);
%     时域
    figure(440)
    subplot(2,1,1);
    plot(t,p,'LineWidth',2,'Color',[0,0,0]);
    
    hold on
    plot(t,p11,'LineWidth',2);
    hold off
    
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %可代替2.1
    h=legend('Y(t)','G(t)');
    set(h,'fontsize',axis_daxiao);
    
    % 频域
    subplot(2,1,2);
    plot(fp1,Pp1,'LineWidth',2,'Color',[0,0,0]);%[0.5,0.5,0.5])
    axis([1 100000 0.00000005 1])
    set(gca,'XScale','log')
    set(gca,'YScale','log')
    
    hold on
    plot(fp11,Pp11,'LineWidth',2);%,'Color',[0,0,0])  
    hold off
    set(gca,'XScale','log') 
    set(gca,'YScale','log') 
               
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('f (Hz)','FontSize',axis_daxiao);ylabel('FFT coefficients','FontSize',axis_daxiao); %可代替2.1
    h=legend('Y(t)','G(t)');
    set(h,'fontsize',axis_daxiao);
   %% 4.6 改用形态学滤波 滤除工频等低频干扰
    g=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ];%101de  34*3+7
    op=1;
    ap1=fushi2(p11,g,op);   %0.804677
    ap1=pengzhang2(ap1,g,op);  %0.781250
%     [fp11_1,Pp11_1]=spectrogram1(length1,p11-ap1);
%     %小波去噪时间   clock记录 0.5740    0.5850     tic记录 0.563437
%     %形态学去噪时间 clock记录 1.6270s   1.6560     tic记录 1.646713   1.624328   1.662696
    %% 形态学滤波重新绘图
    figure(443)
    subplot(2,2,1);
    plot(t,p11,'LineWidth',2,'Color',[0,0,0]);
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman');
    xlabel('t (seconds)','FontSize',axis_daxiao);
    
    hold on
    plot(t,ap1,'LineWidth',2);ylabel('Voltage  (mv)','FontSize',axis_daxiao);
    h=legend('G(t)','L(t)');
    set(h,'FontSize',axis_daxiao);
    hold off

    subplot(2,2,2);
    plot(t,p11-ap1,'LineWidth',2,'Color',[0,0,0]);
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %可代替2.1
    h= legend('H(t)');
    set(h,'FontSize',axis_daxiao);
    % 频域 原始信号
    subplot(2,2,3);
    [fp112,Pp112]=spectrogram1(length1,p11);
    
    plot(fp112,Pp112,'LineWidth',2,'Color',[0,0,0]) 

    xlabel('f (Hz)','FontSize',axis_daxiao)
    ylabel('FFT coefficients','FontSize',axis_daxiao)
    axis([0.1 100000 0.00000005 10])  
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','XScale','log') 
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','YScale','log')
    % 2.01.1分析 小波去基底 后的 频域图
    % 频域 处理后的信号
    subplot(2,2,4);
    
    [f,P1]=spectrogram1(length1,p11-ap1);
    
    plot(f,P1,'LineWidth',2,'Color',[0,0,0]) 
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','XScale','log') 
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','YScale','log')
    xlabel('f (Hz)','FontSize',axis_daxiao)
    ylabel('FFT coefficients','FontSize',axis_daxiao)
    axis([0.1 100000 0.00000005 10])  
    set(gcf,'unit','normalized','position',[0,0,1,0.6]);
    
    %% 细节放大
    p11ap=p11-ap1;
    figure(444)
    
    subplot(2,1,1);
    plot(t,p11,'LineWidth',2,'Color',[0,0,0]);
    hold on
    plot(t,ap1,'LineWidth',2);%,'Color',[0,0,0]);
        
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %可代替2.1
    h=legend('G(t)','L(t)');
    set(h,'FontSize',axis_daxiao);
    hold off
        
    subplot(2,1,2);
    plot(t,p11,'LineWidth',2,'Color',[0,0,0]);
    hold on
    plot(t,p11ap,'LineWidth',2);%,'Color',[0,0,0]);
        
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %可代替2.1
    h=legend('G(t)','H(t)');
    set(h,'FontSize',axis_daxiao);
    hold off
    
   %% %去噪完成，下面是脉冲定位%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 对p21进行 幅度的定位 使用噪声估计方差，再设置阈值，峰值检测，并且剔除了一些伪峰。  
    tic %峰值定位时间
    % 第一部分 判断阈值，找峰值
    daleta_p11ap=median(abs(p11ap))/0.6745; %估计得p21的噪声方差
    r_p11ap=10*daleta_p11ap*sqrt(2*log(length(p11ap))); %计算的是阈值。 13.8    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    [v1,l1]=findpeaks(p11ap,'minpeakheight',10);%找到大于零的所有极值
    pulse_width=200;
    % 删除间隔小于一个脉宽的峰值
    dl1=l1(2:end)-l1(1:end-1);
    xd1=find(dl1 <pulse_width);        %大于一个脉宽，获得
    %记录要删除的点
    a=[];
    for i = fliplr(xd1) %数组反向
%         i
        if l1(i+1)-l1(i)>20
            a(end+1)=i;
            a(end+1)=i+1;
            continue
        end
        if v1(i)>v1(i+1) %比较峰值大小
            a(end+1)=i+1;
        else
            a(end+1)=i;
        end
    end
    a=sort(a);
    b=find(diff(a)==0);
    a(b)=[];

    for j = fliplr(a)
        l1(j)=[];
        v1(j)=[];
    end
    
    %  删除较高的脉冲极值
    l11=find(v1>120);
    for i = fliplr(l11)
        l1(i)=[];
        v1(i)=[];
    end
    % 5.3脉冲的选取
    pulse_width=200;%20KHz,一个脉冲2ms计算，2*20

    length_l=length(l1);%l为极值点的横坐标，一共检索185个脉冲,后来是211
    pulse=zeros(length_l,pulse_width);
    for i =length_l:-1:1
        pulse(i,:)=p11ap((l1(i)-pulse_width/2 : l1(i)+pulse_width/2-1));
    end
    
    for i =length(l1):-1:1
        if find(pulse(i,:) < -r_p11ap*0.5) %去除有重叠的双峰信号,是峰值的多少倍
            pulse(i,:)=[];
            l1(i)=[];
            v1(i)=[];
        end
    end
        
    pulse(all(pulse==0,2),:)=[]; %删除全零行
    % 将脉冲都拉为零
    for i = size(pulse,1):-1:1
%        i
       if max(pulse(i,:)) > v1(i)  %35 %拉低以后，设置阈值
           pulse(i,:)=[];
           l1(i)=[];
           v1(i)=[];
       end
    end
    
    for i = size(pulse,1):-1:1
       pulse(i,:)=pulse(i,:)-pulse(i,1);
    end
    
    for i =length(l1):-1:1
        if find(pulse(i,:) < -3) %去除有重叠的双峰信，阈值因信号而定
            pulse(i,:)=[];
            l1(i)=[];
            v1(i)=[];
        end
    end
    
    for i =length(l1):-1:1
        if max(pulse(i,:)) < 12 %去除有重叠的双峰信号，阈值因信号而定
            pulse(i,:)=[];
            l1(i)=[];
            v1(i)=[];
        end
    end
    % 绘制峰值点
    figure(501)
    plot(t,p11ap,'LineWidth',2,'Color',[0,0,0]);
    hold on
    sz = 500;
    scatter((l1-1)*T,v1,sz,'r.');
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %可代替2.1
%     toc  %峰值定位 0.186772s   0.135832s   0.122593s   0.195461s
    %% 5.4随机绘制六个脉冲图像
    j =0;
    for i=17:1:22 %可调整
        j=j+1;
        figure(522);
        subplot(3,2,j);
        plot(pulse(i,:),'LineWidth',2,'Color',[0,0,0]);%ylabel('Intercepted pulse');xlabel('Datapoints');

        axis([0,pulse_width,0,80]);
        set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
        xlabel('Datapoints','FontSize',axis_daxiao);ylabel('Voltage (mv)','FontSize',axis_daxiao-3);
    end
    %% 归一化每个脉冲，分成两个。后期需要归一化成一个。
    pulse_2=[];
    Pulse_number=size(pulse,1); 
    for i=1:Pulse_number
        if max(pulse(i,:))>35
            pulse_2(:,i)=pulse(i,:)./max(pulse(i,:))*53;
        else
            pulse_2(:,i)=pulse(i,:)./max(pulse(i,:))*22;
        end
    end
    pulse_1 = pulse_2(:,:)';
    %归一化时长0.002132s
    %% 6.0提取的脉冲信号 定义所有特征
    I6=sum(pulse_1,2)';%208*1的列向量  %I5面积
    Pulse_number=size(pulse_1,1); %208
    %
    remenber=[];
    figure(123)
    for i=1:Pulse_number    %208是一个伪脉冲
%         i
        I1(i)=max(pulse(i,:));   %幅值
        if max(pulse(i,:))>35
            axy1 =plot(pulse_1(i,:),'LineWidth',2,'Color',[0 0 1]);           
        else
            axy2 =plot(pulse_1(i,:),'LineWidth',2,'Color',[1 0 0]);
        end
  
        hold on
        x1 = find(pulse_1(i,:) > (max(pulse_1(i,:))/2));
        I2(i)=length(x1); %I2=Width=fwhm 的点数
        
        x1 = find(pulse_1(i,:) > (max(pulse_1(i,:))*0.2));
        I3(i)=length(x1); %I2  0.2高度的点数

        x2 = find(pulse_1(i,:) > (max(pulse_1(i,:))*0.1)); %找到底部的数据点集
        x3 = find(pulse_1(i,:) > (max(pulse_1(i,:))*0.9)); %找到顶部的数据点集

        I4(i) = x3(1)-x2(1); %上升时间的点数
        
        I5(i) = x2(end)-x3(end);%下降时间的点数

        I7(i) = (max(pulse_1(i,:))- pulse_1(i,x2(1)))   / (I3(i));%I6是上升峭度
        I8(i) = (max(pulse_1(i,:))- pulse_1(i,x3(end))) / (I4(i));%I7是下降峭度
        
        %每个脉冲的 幅度归一化后的斜率最大值
        I9(i)=max(diff(pulse_1(i,:))); 
        
        I10(i) = mean(pulse_1(i,x2)); %这里是底部，就是平均值

        I11(i) = std(pulse_1(i,x2));

        I12(i) = skewness(pulse_1(i,x2)); %偏度,左偏小于零;

        I13(i) = kurtosis(pulse_1(i,x2)); %峰度,

        %I14,频域变换后的积分
        L=pulse_width;        % 信号长度
        Fs = 200000;          % 采样频率
        T = 1/Fs;             % 采样周期      
        t1 = (0:L-1)*T;        % 时间向量!  296913个/200000=1.3S时长
        f = Fs*(0:(L/2))/L;

        PULSE_Y = fft(pulse_1(i,:));  
        PY2 = abs(PULSE_Y/L);
        PY1 = PY2(1:L/2+1);
        PY1(2:end-1) = 2*PY1(2:end-1);
        P(i,:)=PY1;

        I14(i) = sum(P(i,:));%变换积分

        %I15变换子标准差
        I15(i)=std(P(i,:));

        %I16变换偏度
        I16(i)=skewness(P(i,:));

        %I17变换峰度
        I17(i)=kurtosis(P(i,:));

    end
    hold off
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('Datapoints','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %可代替2.1

    h=legend([axy1(1),axy2(1)],'Particle pulses standardized to 80 μm size','Particle pulses standardized to 57.9 μm size');
    set(h,'FontSize',axis_daxiao-5);
    %
    for i=1:Pulse_number    %208是一个伪脉冲
        [pc,pl]=wavedec(pulse_1(i,:),8,'coif3');%分解
        %重构第1~5层逼近信号 低频主要是为了低频去除基底
        Pa5=wrcoef('a',pc,pl,'coif3',5);%  查询wavemngr('read',1)，haar，
        Pa4=wrcoef('a',pc,pl,'coif3',4);
        Pa3=wrcoef('a',pc,pl,'coif3',3);
        Pa2=wrcoef('a',pc,pl,'coif3',2);
        Pa1=wrcoef('a',pc,pl,'coif3',1);

        Pd5=wrcoef('d',pc,pl,'coif3',5);
        Pd4=wrcoef('d',pc,pl,'coif3',4);
        Pd3=wrcoef('d',pc,pl,'coif3',3);
        Pd2=wrcoef('d',pc,pl,'coif3',2);
        Pd1=wrcoef('d',pc,pl,'coif3',1);

        EPa3(i)=norm(Pa3,2); % 小波分解的低频能量大小
        EPa2(i)=norm(Pa2,2);
        EPa1(i)=norm(Pa1,2);
        
        I18(i)=EPa3(i);
        I19(i)=norm(Pd3,2); % 小波分解的中频能量大小
        I20(i)=norm(Pd4,2); % 小波分解的中频能量大小
        I21(i)=norm(Pd5,2); % 小波分解的中频能量大小
    end
    %特征提取时间很少 0.659168 秒  42个脉冲！！
    %% 9.0表格输出，调整变量格式
    I1_mv=I1';
    I2_FWHM_dian=I2';
    I3_02=I3';
    I4_rising_time=I4';
    I5_droping_time=I5';
    I6_area=I6';
    I7_rising_slope=I7';
    I8_falling_slope=I8';
    I9_max_slope=I9';
    I10_mean=I10';
    I11_psd=I11';
    I12_pulse_skewness=I12';
    I13_p_kurtosis=I13';
    I14_FFT_mean=I14'; %变换积分
    I15_FFT_SSD=I15'; %变换标准差
    I16_T_skewness=I16'; %偏度
    I17_T_kurtosis=I17';
    I18_EPa3=I18';
    I19_EPd3=I19';
    I20_EPd4=I20';
    I21_EPd5=I21';
    toc
    %%    % 存储全部特征
    T = table(I1_mv,I2_FWHM_dian,I3_02,I4_rising_time,I5_droping_time,I6_area,...
              I7_rising_slope,I8_falling_slope,I9_max_slope,I10_mean,I11_psd,I12_pulse_skewness,...
              I13_p_kurtosis,I14_FFT_mean,I15_FFT_SSD,I16_T_skewness,I17_T_kurtosis,I18_EPa3,I19_EPd3,I20_EPd4,I21_EPd5);
    filename = doc_Amplitude(ii,:);

    writetable(T,filename);
    %% 9.1输出表格 180个脉冲的序列
    T1 = table(pulse_1'); %这里的脉冲，可以调节上面的阈值，得到更多
    filename1 = doc_pulse(ii,:);
    writetable(T1,filename1);

    %% 分批处理数据，删除原始数据
    %分批处理，请取消下面两行注释
%     close all
%     clear all
    
end







