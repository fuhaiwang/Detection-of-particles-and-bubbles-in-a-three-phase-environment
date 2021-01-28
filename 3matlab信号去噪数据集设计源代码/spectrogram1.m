function[f,P1]=spectrogram1(length1,x)

Fs = 200000;           % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = length1;          % Length of signal
t = (0:L-1)*T;        % Time vector

Y = fft(x);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
% subplot(2,2,2);
% plot(f,P1) 
% % title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('FFT coefficients')
% axis([0.1 10000 0.0000005 1])  
% set(gca,'XScale','log') 
% set(gca,'YScale','log') 
end