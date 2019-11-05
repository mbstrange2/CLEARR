%% Author: Dante Muratore - dantemur@stanford.edu
% 05-Feb-2018
% -- based on previous work from Nathan Staffa
%
% This script provides basic Matlab interaction to .bin files from
% retina recordings in the Chichilnisky lab. It uses the spike sorter
% Vision to support some of the functionalities.

% The script reads .bin files, calibrate the offset of each channel and
% saves the output in a .mat file.

clear all;close all;clc; 

%% Set paths -- this path is user dependant %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function
dataPath = '/home/ubuntu/data/2013-05-28-4/data000/';
outputPath = '/home/ubuntu/data/2013-05-28-4/mat/';
util = '/home/ubuntu/vision7-unix/';
spikePath='/home/ubuntu/analysis/2013-05-28-4/data000/data000.spikes';
spikeFile=edu.ucsc.neurobiology.vision.io.SpikeFile(spikePath);
setPaths(dataPath,outputPath,util)
%% Set global parameter
fs = 20e3;
Tmeas = 6*60;
nSamplesToRead = Tmeas*fs;   % how many samples to read
startSample = 0;
bufferSize = 100000; %1000000; % The number of samples to process at a time.
% The number of samples processed before starting a new .bin file
samplesPerBinFile = 2.4e6;  % 2.4e6 to limit to <2GB per file.

%% Start parallel pool
p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    parpool(2)                 % leave 12 cores available
    nw = 2;
    fprintf(['Local cluster built with ', num2str(nw), ' workers.\n'])
 else
     nw = p.NumWorkers;
     fprintf(['Local cluster already exists with ', num2str(nw), ' workers.\n'])
 end
%% Get the raw file handler object
disp('=============================== Initializing Algorithms ===============================');
% Gets raw data and sets header
[rawFile, header, totalSamples, nElec] = getRawData(dataPath, nSamplesToRead, startSample);
%% Runs readout + decoder on 100'000 samples at a time
fprintf([datestr(now, 'HH:MM:SS'),'\n -- Processing data from: ', dataPath,...
    '\n -- Saving data to: ', outputPath, '\n']);
tic;
% Stores the number of samples remaining to be copied.
nSamplesToCopy = nSamplesToRead; % So it resets each time.
% Keeps track of the current file's # of samples to allow of ensuring that
% the individual files stay under 2GB (in case of old FAT filesystems).
sampsThisFile = 0;
firstCycle = 1;
% start with empty matrix
newData = zeros(nSamplesToRead,nElec+1);

while nSamplesToCopy > 0
    
    sampsThisFile = sampsThisFile + min(bufferSize,nSamplesToCopy);
    
    % either fills the buffer entirely, or grabs the remaining samples
    rawData = rawFile.getData(startSample, min(bufferSize,nSamplesToCopy));
    nSamples = size(rawData,1);
    if firstCycle
        % gets an offset per channel in a vector long nElec
        samplesForOffset = min(nSamples,50e3);
        [offsetPerChannel] = offsetPerChannel_mex(rawData(1:samplesForOffset,2:end));
        firstCycle = 0;
    end
    
    % define the tmp array and save calibrated data to it
    tmp = zeros(nSamples,nElec+1,'int16');
    parfor i=1:nSamples
        [tmp(i,:)] = offsetCalibration(rawData(i,:),offsetPerChannel);
    end
   
    % Appends data passed in to the final matrix
    newData(startSample + 1 : startSample + size(tmp,1),:) = tmp;
    % You've copied a full buffer so you now need to copy less.
    nSamplesToCopy = nSamplesToCopy - bufferSize;
    startSample = startSample + bufferSize; % also increase your start index.
    disp(['====================== samples left to copy: ', num2str(nSamplesToCopy), ' ======================']);
end

disp([datestr(now, 'HH:MM:SS'),' -- Finished processing']);
toc;

% close the internal arrays to free memory.
rawFile.close;

%outputFile = [outputPath '/data000_' num2str(Tmeas) 's.mat'];
%save(outputFile, 'newData', '-v7.3');

 for ArrayIndex=1:1:512
     Tspike=spikeFile.getSpikeTimes(ArrayIndex);
     SpikeCount=size(Tspike);
     SpikeCount=SpikeCount(1,1);
%     for i=1:1:SpikeCount
     for i=1:1:min(30,SpikeCount)
         t=Tspike(i);
         temp=newData(t:t+60,ArrayIndex);
         
         SpikeClip=[outputPath '/data000_electrode' num2str(ArrayIndex) 'spike' num2str(i) '.mat'];
         save(SpikeClip, 'temp');
     end
 end
 
 
 
%%

%comments: channel 412 has a nice spike as example
% figure(1)
% t = 1:1:size(rawData,1);
% tNew = size(newData,1) - size(rawData,1) + 1 : 1 : size(newData,1);
% nch = 1;
% %t = 550:1:700;
% plot(t,rawData(t,nch),t,newData(tNew:end,nch));
% 
% xlabel('sample');ylabel('code');
% grid on
% lg = legend('original','reconstructed','Location', 'southwest');
% set(lg,'FontSize',12)
% set(gcf,'color','w')
%  nyquistRate = B*20e3*nElec;
% mCollisions = mean(nCollisions)
% mActivityFree = mean(activityFactor)*20e3
% compressionRatio = nyquistRate/mActivityFree
