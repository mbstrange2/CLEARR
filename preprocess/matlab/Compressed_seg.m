%% Compressed data segmentation
clear all;close all;clc; 

%% Set paths -- this path is user dependant %
% utilities, data and output paths are passed
% Vision paths are defined in setPaths function
dataPath = '/home/ubuntu/data/2013-05-28-4/data000/';
outputPath = '/home/ubuntu/data/2013-05-28-4/mat/';
util = '/home/ubuntu/vision7-unix/';
spikePath='/home/ubuntu/analysis/2013-05-28-4/data000/data000.spikes';
setPaths(dataPath,outputPath,util);
spikeFile=edu.ucsc.neurobiology.vision.io.SpikeFile(spikePath);
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
