%% Author: Dante Muratore - dantemur@stanford.edu
% 05-Feb-2018
% -- based on previous work from Nathan Staffa
%
% This function provides basic Matlab interaction to .bin files from
% retina recordings in the Chichilnisky lab. It uses the spike sorter
% Vision to support some of the functionalities.

clear all;close all;clc; 

%% Set paths -- this path is user dependant %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function
dataPath = '/sim/dantemur/readout/data/2015-11-09-3/orig/data000/';
outputPath = '/sim/dantemur/readout/data/2015-11-09-3/tmp/';    % useless - kept only to use setPaths
util = '/sim/dantemur/readout/utilities/';
setPaths(dataPath,outputPath,util)
setPaths(dataPath,outputPath,util)

%% Set global parameters
fs = 20e3;
Tmeas = 0.1;                  % measure 1 second 
nSamplesToRead = Tmeas*fs;   % how many samples to read
startSample = 0;
bufferSize = 20000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.
B = single(12);         % B bits for quantizer
FS = single(2048);      % original full scale
FSeff = single(256);       % adpated to maximum signal
Beff = single(B - ceil(log2(FS/FSeff)));    % effective number of bits

%% Start parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    parpool(16)                 % leave 12 cores available
    nw = 16;
    fprintf(['Local cluster built with ', num2str(nw), ' workers.\n'])
else
    nw = p.NumWorkers;
    fprintf(['Local cluster already exists with ', num2str(nw), ' workers.\n'])
end
%% Get the raw file handler object and electrode map
disp('=============================== Initializing Algorithms ===============================');
% Gets raw data and sets header
[rawFile, header, totalSamples, nElec] = getRawData(dataPath, nSamplesToRead, startSample);
% Gets electrode map to move from 512 vector to 16x32 array
getMap
%% Runs readout + decoder on 100'000 samples at a time
fprintf([datestr(now, 'HH:MM:SS'),'\n -- Processing data from: ', dataPath, '\n']);
tic;
% Stores the number of samples remaining to be copied.
nSamplesToCopy = nSamplesToRead; % So it resets each time.

while nSamplesToCopy > 0
    
    % either fills the buffer entirely, or grabs the remaining samples
    rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    nSamples = size(rawData,1);
    % gets a offset per channel in a vector long nElec
    samplesForOffset = min(nSamples,50e3);
    [offsetPerChannel] = offsetPerChannel_mex(rawData(1:samplesForOffset,2:end));
    
     % define newData matrix
    nwire            = 1:1:16;
    MSE_newData      = zeros(size(nwire));
    newData          = zeros(nSamples,nElec+1,'int16');
    rawDataNoOffset  = zeros(nSamples,nElec+1,'int16');
    bitTx            = zeros(nSamples,size(nwire,2),'single');
    
    parfor i=1:nSamples
        % calibrate offset from rawData to calculate MSE
        [rawDataNoOffset(i,:)] = offsetCalibration(rawData(i,:), offsetPerChannel)
    end
    
    for j = 1:size(nwire,2)
        % Samples are run in parallel for speed
        w = nwire(j);
        parfor i=1:nSamples
            [newData(i,:), bitTx(i,j)] = ramp_adc_flag_NC_param(rawData(i,:),offsetPerChannel...
                ,posLin,Beff,FSeff,w);
        end
        MSE_newData(j) = immse(rawDataNoOffset,newData);
    end
    
    % You've copied a full buffer so you now need to copy less.
    nSamplesToCopy = nSamplesToCopy - bufferSize;
    startSample = startSample + bufferSize; % also increase your start index.
    disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
end

disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
toc;

rawFile.close;


% MSE_newData(2) = immse(rawDataNoOffset,newData(:,:,2));
% MSE_newData(3) = immse(rawDataNoOffset,newData(:,:,3));
% MSE_newData(4) = immse(rawDataNoOffset,newData(:,:,4));
% MSE_power = immse(rawDataNoOffset,zeros(size(rawDataNoOffset),'int16'));
% MSE_newData/MSE_power

%%
%comments: channel 412 has a nice spike as example
% load('/sim/dantemur/readout/data/2015-11-09-3/tmp/wires.mat')
figure(1)
t = 1:1:size(rawData,1);
nch = 412;
ref = ones(size(t))*single(offsetPerChannel(nch-1));
%t = 550:1:700;
plot(t,rawData(t,nch)-offsetPerChannel(nch-1),t,newData(t,nch), t, ref);
xlabel('sample');ylabel('code');
grid on
lg = legend('original','single-wire','double-wire','four-wire','Location', 'southwest');
set(lg,'FontSize',12)
set(gcf,'color','w')

% bitTxMean = mean(bitTx)
% bitTxMin = min(bitTx)
% bitTxMax = max(bitTx)
% bitTxSig = std(bitTx)
% compressionFactorMean = B*nElec./bitTxMean

% figure(2)
% sampleVector = 1:20:nSamples;
% scatter(t(sampleVector),bitTx(sampleVector,1))
% hold on
% scatter(t(sampleVector),bitTx(sampleVector,2))
% scatter(t(sampleVector),bitTx(sampleVector,3))
% xlabel('Sample')
% ylabel('Transmitted bits per sample')
% legend('#wire = 1', '#wire = 2', '#wire = 4')
% ylim([(min(bitTxMean) - 3*min(bitTxSig)) (max(bitTxMean) + 3*max(bitTxSig))])
% hold off
% 

figure(2)
yyaxis left
plot(nwire,MSE_newData)
xlabel('# wires')
ylabel('MSE')
grid on
yyaxis right
plot(nwire,(B*nElec)./mean(bitTx))
ylabel('Compression')
legend('MSE','Compression','Std Rate')

