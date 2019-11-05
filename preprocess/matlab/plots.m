

%% Number of ghost per samples as a function of number of bits
figure(1)
b = 8:1:12;
m = [510,510,509,506,502;508,505,499,486,462;505,499,485,460,396];
plot(b,m)
grid on
xlabel('Quantizer bits')
ylabel('# ghosts per sample')
ylim([min(min(m)) 512])
legend ('nw = 1', 'nw = 2', 'nw = 4', 'Location', 'southwest')
set(gcf,'color','w')

%% System data rate for different strategies
figure(2)
Nch = (0.1:0.1:10)*1e3;
B = 8:1:12;
Nch = Nch';
fs = 15e3;
acol = 0.5;
% no compression
f1 = Nch*B*fs;
% 'lossless' compression
f2 = 2*ceil(sqrt(Nch))*2.^B*fs;
% lossy compression
f3 = 2*ceil(sqrt(Nch))*(1-acol)*2.^B*fs;
% lossy compression sending address
f4 = ceil(log2(Nch))*(1-acol)*2.^B*fs;

i = 1;
f_150M = ones(size(Nch))*150e6;
loglog(Nch,f1(:,i),Nch,f2(:,i),Nch,f3(:,i),Nch,f4(:,i))
lg2 = legend ('none', 'lossless', 'lossy', 'lossy + address',...
    'Location', 'northwest');
set(lg2,'FontSize',12)
hold on
loglog(Nch,f_150M,'--k')
hold off
grid on
xlabel('Nch')
ylabel('f_{TX} [MHz]')
xlim([100 10000])
ylim([1e7 1e9])
set(gcf,'color','w')

%% Distribution of collisions per code
figure(3)
ramp = (-FSeff:2*FSeff/2^Beff:FSeff-1)+0.5;
plot(ramp,mean(simbolsDecoded,1))
grid on
xlabel('Code')
ylabel('# collisions per sample')
xlim([min(ramp) max(ramp)])
set(gcf,'color','w')

%% Distribution of number of collissions
figure(4)
edges = [1, 2, 4, 8, 16, 32, 64, 128, 256];
C = histc(simbolsDecoded,edges,2);
Ctot = sum(C);
bar(Ctot)
set(gca,'XTickLabel',{'1', '2:3', '4:7', '8:15', '16:31', '32:63',...
    '64:127', '128:255', '256:512'}) 
grid on
xlabel('# collisions')
ylabel('counts')
set(gcf,'color','w')
figure(5)
weights = [1, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5, 191.5, 384];
bar(Ctot.*weights)
set(gca,'XTickLabel',{'1', '2:3', '4:7', '8:15', '16:31', '32:63',...
    '64:127', '128:255', '256:512'}) 
grid on
xlabel('# collisions')
ylabel('counts * mean{interval}')
set(gcf,'color','w')

%% Number of transmitted channel as a function of compression factor
figure(6)
expansionFactor = 1:1:10;
bitperChannel = 9;      % 5 per row and 4 per column
fs = 20e3;
index = find(symbolsDecoded<=cF & symbolsDecoded>0);
tot = sum(symbolsDecoded(index));
dSA = tot/nSamplesToRead
dataSentAvg = zeros(10,1);
dataSentAvg(1)  = 16.14;
dataSentAvg(2)  = 17.68;
dataSentAvg(3)  = 19.48;
dataSentAvg(4)  = 46.38;
dataSentAvg(5)  = 46.38;
dataSentAvg(6)  = 59.03;
dataSentAvg(7)  = 59.03;
dataSentAvg(8)  = 61.13;
dataSentAvg(9)  = 90.63;
dataSentAvg(10) = 92.93;
compressionRatio = nElec*B./(dataSentAvg.*bitperChannel);
[hAx]=plotyy(expansionFactor,dataSentAvg.*bitperChannel*fs/1e6,expansionFactor,compressionRatio)
grid on
xlabel('Expansion Factor')
ylabel(hAx(1),'System Data Rate [Mbps]')
ylabel(hAx(2),'Compression ratio')
set(gcf,'color','w')















