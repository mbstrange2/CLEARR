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
dataPath = '/sim/dantemur/readout/data/2005-05-26-7/orig/data006/';
outputPath = '/sim/dantemur/readout/data/2005-05-26-7/flag_NC/data006';
outputPath2 = '/sim/dantemur/readout/data/2005-05-26-7/flag_NC_2/data006';
outputPath4 = '/sim/dantemur/readout/data/2005-05-26-7/flag_NC_4/data006';
util = '/sim/dantemur/readout/utilities/';
setPaths(dataPath,outputPath,util)
setPaths(dataPath,outputPath4,util)

%% Set global parameters
fs = 20e3;
Tmeas = 60*6;
nSamplesToRead = Tmeas*fs;   % how many samples to read
startSample = 0;
bufferSize = 1000000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.
B = single(10);         % B bits for quantizer
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
fprintf([datestr(now, 'HH:MM:SS'),'\n -- Processing data from: ', dataPath,...
    '\n -- Saving data to: ', outputPath, '\n']);
tic;
% Stores the number of samples remaining to be copied.
nSamplesToCopy = nSamplesToRead; % So it resets each time.
% Create the new file (Creates dataxyz000.bin and copies the header info)
newFile = edu.ucsc.neurobiology.vision.io.ModifyRawDataFile(outputPath, header);
newFile2 = edu.ucsc.neurobiology.vision.io.ModifyRawDataFile(outputPath2, header);
newFile4 = edu.ucsc.neurobiology.vision.io.ModifyRawDataFile(outputPath4, header);
% Keeps track of the current file's # of samples to allow of ensuring that
% the individual files stay under 2GB (in case of old FAT filesystems).
sampsThisFile = bufferSize;

while nSamplesToCopy > 0
    
    sampsThisFile = sampsThisFile + min(bufferSize,nSamplesToCopy);
    if (sampsThisFile > samplesPerBinFile) % We're at the size limit
        newFile.addFile; % Start a new .bin file
        newFile2.addFile; % Start a new .bin file
        newFile4.addFile; % Start a new .bin file
        sampsThisFile = 0;
    end
    
    % either fills the buffer entirely, or grabs the remaining samples
    rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    nSamples = size(rawData,1);
    % gets a offset per channel in a vector long nElec
    samplesForOffset = min(nSamples,50e3);
    [offsetPerChannel] = offsetPerChannel_mex(rawData(1:samplesForOffset,2:end));
    
     % define newData matrix
    newData     = zeros(nSamples,nElec+1,'int16');
    newDataAvg  = zeros(nSamples,nElec+1,'int16');
    newData2     = zeros(nSamples,nElec+1,'int16');
    newDataAvg2  = zeros(nSamples,nElec+1,'int16');
    newData4     = zeros(nSamples,nElec+1,'int16');
    newDataAvg4  = zeros(nSamples,nElec+1,'int16');

    % Samples are run in parallel for speed
    parfor i=1:nSamples
        [newData(i,:)] = ramp_adc_flag_NC_mex(rawData(i,:),offsetPerChannel...
            ,posLin,Beff,FSeff,single(1));
        [newData2(i,:)] = ramp_adc_flag_NC_mex(rawData(i,:),offsetPerChannel...
            ,posLin,Beff,FSeff,single(2));
        [newData4(i,:)] = ramp_adc_flag_NC_mex(rawData(i,:),offsetPerChannel...
            ,posLin,Beff,FSeff,single(4));
   end
    
    parfor k = 2:nElec+1
        % interpolates values to fill in not decoded samples
        newDataAvg(:,k)  = neuralDataAveraging_mex(newData(:,k),int16(5));
        newDataAvg2(:,k) = neuralDataAveraging_mex(newData2(:,k),int16(5));
        newDataAvg4(:,k) = neuralDataAveraging_mex(newData4(:,k),int16(5));
    end
    % copies synchronization channel
    newDataAvg(:,1) = rawData(:,1);
    newDataAvg2(:,1) = rawData(:,1);
    newDataAvg4(:,1) = rawData(:,1);

    % Appends data passed in to the end of final .bin file in the folder
    newFile.appendDataToLastFile(newDataAvg);
    newFile2.appendDataToLastFile(newDataAvg2);
    newFile4.appendDataToLastFile(newDataAvg4);
    % You've copied a full buffer so you now need to copy less.
    nSamplesToCopy = nSamplesToCopy - bufferSize;
    startSample = startSample + bufferSize; % also increase your start index.
    disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
end

disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
toc;

newFile.close; % close the internal arrays to free memory.
newFile2.close; 
newFile4.close;
rawFile.close;


%%
%comments: channel 412 has a nice spike as example
% load('/sim/dantemur/readout/data/2015-11-09-3/tmp/wires.mat')
figure(1)
t = 1:1:size(rawData,1);
nch = 412;
ref = ones(size(t))*single(offsetPerChannel(nch+1));
%t = 550:1:700;
plot(t,rawData(t,nch)-offsetPerChannel(nch-1),t,newData2(t,nch),t,newData4(t,nch));
xlabel('sample');ylabel('code');
grid on
lg = legend('original','single-wire','double-wire','four-wire','Location', 'southwest');
set(lg,'FontSize',12)
set(gcf,'color','w')
