%�� %����˼·��С����ֵ��ȥ����Ƶ������̬ѧȥ����Ƶ������ȡ�����������档
%help wavelet %wavedemo
%wavemenu С�������� %https://ww2.mathworks.cn/help/deeplearning/signal-processing-using-deep-learning.html���ѧϰ��matlab����
close all  
clear all  %����106,����70,80,���126,148

for ii = 50:50 %������  70-100����  105-124����  125-148���   % ��ȡԭʼ���ݣ������
    ii
    doc_Amplitude(ii,:) =['C:\Users\fuhai\Desktop\8-13\��һ��\Amplitude813_',num2str(ii),'.xlsx']; %���Ϊ�Լ��Ĵ洢·��
    doc_pulse(ii,:)=['C:\Users\fuhai\Desktop\8-13\��һ��\pulse10-8_',num2str(ii),'.xlsx',]; %���Ϊ�Լ��Ĵ洢·��
    %% ��������
    [num1]= xlsread(['C:\Users\fuhai\Desktop\8-13\Test813_',num2str(ii),'.xlsx'],2,'B3:B1048576');  %���Ϊ�Լ��Ķ�ȡ·��
    [num2]= xlsread(['C:\Users\fuhai\Desktop\8-13\Test813_',num2str(ii),'.xlsx'],1,'B3:B954428');   %���Ϊ�Լ��Ķ�ȡ·��
    num = [num1;num2];  %1048574 + 171010 = 1219584
    Alq=num';
    x=1000*Alq;%����������1K����mv��λ�ÿ�
    length1=size(Alq,2);%1219584
    tic
    %% 1.0ԭʼ���ݻ���
    axis_daxiao=38;       %�����С
    Fs = 200000;          % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = length1;          % Length of signal
    t = (0:L-1)*T;        % Time vector
    % %ͼһ��a��ԭʼ�źŵ���ʾ
    figure(2)
    subplot(1,2,1);
    plot(t,x,'LineWidth',2,'Color',[0,0,0])
    set(gca,'FontSize',axis_daxiao-10,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao) %ʱ�� (s)
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

    %% 2.0С���ֽ⣬W = wavenames('all')  bior4.4,db10,3,4��Ҫ��  �����ʵģ�sym2��coif1
    % С��ϸ�ڣ���Ҫ���� ��Ƶȥ��
    [c2,l2]=wavedec(x,4,'db10'); %�ع���1~4���ϸ���ź�
%     pd5=wrcoef('d',c2,l2,'db10',5);
    pd4=wrcoef('d',c2,l2,'db10',4);
    pd3=wrcoef('d',c2,l2,'db10',3);
    pd2=wrcoef('d',c2,l2,'db10',2);
    pd1=wrcoef('d',c2,l2,'db10',1);

    %% 2.1С����ֵ�� ����������
    %��ֵ����ȥ����Ƶ����.
    %Ŀǰ�ж�����ֵѡȡ�ķ������̶���ֵ����sqtwolog��minimaxi����ֵ��ֵ���ơ���ƫ��Ȼ����rigrsure�Լ�����ʽ���Ƶȣ�NΪ�źų��ȣ�heursure  ��ֵ������ѡ�����Ҳ�кö���
    %��С��ϸ��ϵ������С������ֵ�˲�����ȷ����ֵ���У�ͳһ��ֵ����ƫ������ֵ������ʽ��ֵ��MiniMax��ֵ��
    r1=median(abs(pd1))/0.6745*sqrt(2*log(length(pd1))); %var()�󷽲�
    r2=median(abs(pd2))/0.6745*sqrt(2*log(length(pd2)));
    r3=median(abs(pd3))/0.6745*sqrt(2*log(length(pd3)));
    r4=median(abs(pd4))/0.6745*sqrt(2*log(length(pd4)));
    %% 2.2�Ľ�����ֵ����.ϵ��Խ�󣬸���ԽС
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
    %% 2.3 ��ֵȥ����ʾ�źŽ��
    p11=x-pd1-pd2-pd3+cd31-pd4+cd41; %+cd11+cd21;%+cd11;  ��Ƶȥ�����
    p=x;
    %% 4.42С����ֵȥ�� �ټ���Ƶ���źŻ�ͼ
    [fp11,Pp11]=spectrogram1(length1,p11); %д��һ�������ˣ�ֱ�Ӹ���FFT��ĵ���Ƶ��ͼ
    [fp1,Pp1]=spectrogram1(length1,p);
%     ʱ��
    figure(440)
    subplot(2,1,1);
    plot(t,p,'LineWidth',2,'Color',[0,0,0]);
    
    hold on
    plot(t,p11,'LineWidth',2);
    hold off
    
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %�ɴ���2.1
    h=legend('Y(t)','G(t)');
    set(h,'fontsize',axis_daxiao);
    
    % Ƶ��
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
    xlabel('f (Hz)','FontSize',axis_daxiao);ylabel('FFT coefficients','FontSize',axis_daxiao); %�ɴ���2.1
    h=legend('Y(t)','G(t)');
    set(h,'fontsize',axis_daxiao);
   %% 4.6 ������̬ѧ�˲� �˳���Ƶ�ȵ�Ƶ����
    g=[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...
       1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ];%101de  34*3+7
    op=1;
    ap1=fushi2(p11,g,op);   %0.804677
    ap1=pengzhang2(ap1,g,op);  %0.781250
%     [fp11_1,Pp11_1]=spectrogram1(length1,p11-ap1);
%     %С��ȥ��ʱ��   clock��¼ 0.5740    0.5850     tic��¼ 0.563437
%     %��̬ѧȥ��ʱ�� clock��¼ 1.6270s   1.6560     tic��¼ 1.646713   1.624328   1.662696
    %% ��̬ѧ�˲����»�ͼ
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
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %�ɴ���2.1
    h= legend('H(t)');
    set(h,'FontSize',axis_daxiao);
    % Ƶ�� ԭʼ�ź�
    subplot(2,2,3);
    [fp112,Pp112]=spectrogram1(length1,p11);
    
    plot(fp112,Pp112,'LineWidth',2,'Color',[0,0,0]) 

    xlabel('f (Hz)','FontSize',axis_daxiao)
    ylabel('FFT coefficients','FontSize',axis_daxiao)
    axis([0.1 100000 0.00000005 10])  
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','XScale','log') 
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','YScale','log')
    % 2.01.1���� С��ȥ���� ��� Ƶ��ͼ
    % Ƶ�� �������ź�
    subplot(2,2,4);
    
    [f,P1]=spectrogram1(length1,p11-ap1);
    
    plot(f,P1,'LineWidth',2,'Color',[0,0,0]) 
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','XScale','log') 
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman','YScale','log')
    xlabel('f (Hz)','FontSize',axis_daxiao)
    ylabel('FFT coefficients','FontSize',axis_daxiao)
    axis([0.1 100000 0.00000005 10])  
    set(gcf,'unit','normalized','position',[0,0,1,0.6]);
    
    %% ϸ�ڷŴ�
    p11ap=p11-ap1;
    figure(444)
    
    subplot(2,1,1);
    plot(t,p11,'LineWidth',2,'Color',[0,0,0]);
    hold on
    plot(t,ap1,'LineWidth',2);%,'Color',[0,0,0]);
        
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %�ɴ���2.1
    h=legend('G(t)','L(t)');
    set(h,'FontSize',axis_daxiao);
    hold off
        
    subplot(2,1,2);
    plot(t,p11,'LineWidth',2,'Color',[0,0,0]);
    hold on
    plot(t,p11ap,'LineWidth',2);%,'Color',[0,0,0]);
        
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %�ɴ���2.1
    h=legend('G(t)','H(t)');
    set(h,'FontSize',axis_daxiao);
    hold off
    
   %% %ȥ����ɣ����������嶨λ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ��p21���� ���ȵĶ�λ ʹ���������Ʒ����������ֵ����ֵ��⣬�����޳���һЩα�塣  
    tic %��ֵ��λʱ��
    % ��һ���� �ж���ֵ���ҷ�ֵ
    daleta_p11ap=median(abs(p11ap))/0.6745; %���Ƶ�p21����������
    r_p11ap=10*daleta_p11ap*sqrt(2*log(length(p11ap))); %���������ֵ�� 13.8    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    [v1,l1]=findpeaks(p11ap,'minpeakheight',10);%�ҵ�����������м�ֵ
    pulse_width=200;
    % ɾ�����С��һ������ķ�ֵ
    dl1=l1(2:end)-l1(1:end-1);
    xd1=find(dl1 <pulse_width);        %����һ���������
    %��¼Ҫɾ���ĵ�
    a=[];
    for i = fliplr(xd1) %���鷴��
%         i
        if l1(i+1)-l1(i)>20
            a(end+1)=i;
            a(end+1)=i+1;
            continue
        end
        if v1(i)>v1(i+1) %�ȽϷ�ֵ��С
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
    
    %  ɾ���ϸߵ����弫ֵ
    l11=find(v1>120);
    for i = fliplr(l11)
        l1(i)=[];
        v1(i)=[];
    end
    % 5.3�����ѡȡ
    pulse_width=200;%20KHz,һ������2ms���㣬2*20

    length_l=length(l1);%lΪ��ֵ��ĺ����꣬һ������185������,������211
    pulse=zeros(length_l,pulse_width);
    for i =length_l:-1:1
        pulse(i,:)=p11ap((l1(i)-pulse_width/2 : l1(i)+pulse_width/2-1));
    end
    
    for i =length(l1):-1:1
        if find(pulse(i,:) < -r_p11ap*0.5) %ȥ�����ص���˫���ź�,�Ƿ�ֵ�Ķ��ٱ�
            pulse(i,:)=[];
            l1(i)=[];
            v1(i)=[];
        end
    end
        
    pulse(all(pulse==0,2),:)=[]; %ɾ��ȫ����
    % �����嶼��Ϊ��
    for i = size(pulse,1):-1:1
%        i
       if max(pulse(i,:)) > v1(i)  %35 %�����Ժ�������ֵ
           pulse(i,:)=[];
           l1(i)=[];
           v1(i)=[];
       end
    end
    
    for i = size(pulse,1):-1:1
       pulse(i,:)=pulse(i,:)-pulse(i,1);
    end
    
    for i =length(l1):-1:1
        if find(pulse(i,:) < -3) %ȥ�����ص���˫���ţ���ֵ���źŶ���
            pulse(i,:)=[];
            l1(i)=[];
            v1(i)=[];
        end
    end
    
    for i =length(l1):-1:1
        if max(pulse(i,:)) < 12 %ȥ�����ص���˫���źţ���ֵ���źŶ���
            pulse(i,:)=[];
            l1(i)=[];
            v1(i)=[];
        end
    end
    % ���Ʒ�ֵ��
    figure(501)
    plot(t,p11ap,'LineWidth',2,'Color',[0,0,0]);
    hold on
    sz = 500;
    scatter((l1-1)*T,v1,sz,'r.');
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('t (seconds)','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %�ɴ���2.1
%     toc  %��ֵ��λ 0.186772s   0.135832s   0.122593s   0.195461s
    %% 5.4���������������ͼ��
    j =0;
    for i=17:1:22 %�ɵ���
        j=j+1;
        figure(522);
        subplot(3,2,j);
        plot(pulse(i,:),'LineWidth',2,'Color',[0,0,0]);%ylabel('Intercepted pulse');xlabel('Datapoints');

        axis([0,pulse_width,0,80]);
        set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
        xlabel('Datapoints','FontSize',axis_daxiao);ylabel('Voltage (mv)','FontSize',axis_daxiao-3);
    end
    %% ��һ��ÿ�����壬�ֳ�������������Ҫ��һ����һ����
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
    %��һ��ʱ��0.002132s
    %% 6.0��ȡ�������ź� ������������
    I6=sum(pulse_1,2)';%208*1��������  %I5���
    Pulse_number=size(pulse_1,1); %208
    %
    remenber=[];
    figure(123)
    for i=1:Pulse_number    %208��һ��α����
%         i
        I1(i)=max(pulse(i,:));   %��ֵ
        if max(pulse(i,:))>35
            axy1 =plot(pulse_1(i,:),'LineWidth',2,'Color',[0 0 1]);           
        else
            axy2 =plot(pulse_1(i,:),'LineWidth',2,'Color',[1 0 0]);
        end
  
        hold on
        x1 = find(pulse_1(i,:) > (max(pulse_1(i,:))/2));
        I2(i)=length(x1); %I2=Width=fwhm �ĵ���
        
        x1 = find(pulse_1(i,:) > (max(pulse_1(i,:))*0.2));
        I3(i)=length(x1); %I2  0.2�߶ȵĵ���

        x2 = find(pulse_1(i,:) > (max(pulse_1(i,:))*0.1)); %�ҵ��ײ������ݵ㼯
        x3 = find(pulse_1(i,:) > (max(pulse_1(i,:))*0.9)); %�ҵ����������ݵ㼯

        I4(i) = x3(1)-x2(1); %����ʱ��ĵ���
        
        I5(i) = x2(end)-x3(end);%�½�ʱ��ĵ���

        I7(i) = (max(pulse_1(i,:))- pulse_1(i,x2(1)))   / (I3(i));%I6�������Ͷ�
        I8(i) = (max(pulse_1(i,:))- pulse_1(i,x3(end))) / (I4(i));%I7���½��Ͷ�
        
        %ÿ������� ���ȹ�һ�����б�����ֵ
        I9(i)=max(diff(pulse_1(i,:))); 
        
        I10(i) = mean(pulse_1(i,x2)); %�����ǵײ�������ƽ��ֵ

        I11(i) = std(pulse_1(i,x2));

        I12(i) = skewness(pulse_1(i,x2)); %ƫ��,��ƫС����;

        I13(i) = kurtosis(pulse_1(i,x2)); %���,

        %I14,Ƶ��任��Ļ���
        L=pulse_width;        % �źų���
        Fs = 200000;          % ����Ƶ��
        T = 1/Fs;             % ��������      
        t1 = (0:L-1)*T;        % ʱ������!  296913��/200000=1.3Sʱ��
        f = Fs*(0:(L/2))/L;

        PULSE_Y = fft(pulse_1(i,:));  
        PY2 = abs(PULSE_Y/L);
        PY1 = PY2(1:L/2+1);
        PY1(2:end-1) = 2*PY1(2:end-1);
        P(i,:)=PY1;

        I14(i) = sum(P(i,:));%�任����

        %I15�任�ӱ�׼��
        I15(i)=std(P(i,:));

        %I16�任ƫ��
        I16(i)=skewness(P(i,:));

        %I17�任���
        I17(i)=kurtosis(P(i,:));

    end
    hold off
    set(gca,'FontSize',axis_daxiao-8,'Fontname','times new Roman') 
    xlabel('Datapoints','FontSize',axis_daxiao);ylabel('Voltage  (mv)','FontSize',axis_daxiao); %�ɴ���2.1

    h=legend([axy1(1),axy2(1)],'Particle pulses standardized to 80 ��m size','Particle pulses standardized to 57.9 ��m size');
    set(h,'FontSize',axis_daxiao-5);
    %
    for i=1:Pulse_number    %208��һ��α����
        [pc,pl]=wavedec(pulse_1(i,:),8,'coif3');%�ֽ�
        %�ع���1~5��ƽ��ź� ��Ƶ��Ҫ��Ϊ�˵�Ƶȥ������
        Pa5=wrcoef('a',pc,pl,'coif3',5);%  ��ѯwavemngr('read',1)��haar��
        Pa4=wrcoef('a',pc,pl,'coif3',4);
        Pa3=wrcoef('a',pc,pl,'coif3',3);
        Pa2=wrcoef('a',pc,pl,'coif3',2);
        Pa1=wrcoef('a',pc,pl,'coif3',1);

        Pd5=wrcoef('d',pc,pl,'coif3',5);
        Pd4=wrcoef('d',pc,pl,'coif3',4);
        Pd3=wrcoef('d',pc,pl,'coif3',3);
        Pd2=wrcoef('d',pc,pl,'coif3',2);
        Pd1=wrcoef('d',pc,pl,'coif3',1);

        EPa3(i)=norm(Pa3,2); % С���ֽ�ĵ�Ƶ������С
        EPa2(i)=norm(Pa2,2);
        EPa1(i)=norm(Pa1,2);
        
        I18(i)=EPa3(i);
        I19(i)=norm(Pd3,2); % С���ֽ����Ƶ������С
        I20(i)=norm(Pd4,2); % С���ֽ����Ƶ������С
        I21(i)=norm(Pd5,2); % С���ֽ����Ƶ������С
    end
    %������ȡʱ����� 0.659168 ��  42�����壡��
    %% 9.0������������������ʽ
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
    I14_FFT_mean=I14'; %�任����
    I15_FFT_SSD=I15'; %�任��׼��
    I16_T_skewness=I16'; %ƫ��
    I17_T_kurtosis=I17';
    I18_EPa3=I18';
    I19_EPd3=I19';
    I20_EPd4=I20';
    I21_EPd5=I21';
    toc
    %%    % �洢ȫ������
    T = table(I1_mv,I2_FWHM_dian,I3_02,I4_rising_time,I5_droping_time,I6_area,...
              I7_rising_slope,I8_falling_slope,I9_max_slope,I10_mean,I11_psd,I12_pulse_skewness,...
              I13_p_kurtosis,I14_FFT_mean,I15_FFT_SSD,I16_T_skewness,I17_T_kurtosis,I18_EPa3,I19_EPd3,I20_EPd4,I21_EPd5);
    filename = doc_Amplitude(ii,:);

    writetable(T,filename);
    %% 9.1������ 180�����������
    T1 = table(pulse_1'); %��������壬���Ե����������ֵ���õ�����
    filename1 = doc_pulse(ii,:);
    writetable(T1,filename1);

    %% �����������ݣ�ɾ��ԭʼ����
    %����������ȡ����������ע��
%     close all
%     clear all
    
end







