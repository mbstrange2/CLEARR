function [] = writeSpikesStruct(analysisPath,dataPath,writePath,arrayID,...
                                cellID,numSpikes,leftSamples,traceLength);
% Function: writeSpikesStruct
% Usage: writeSpikesStruct(analysisPath,dataPath,writePath,arrayID,cellID,...
%                          numSpikes,leftSamples,traceLength);
% ---------------------------------------------------------------------------
% Function to read in the spikes of a neuron of interest, and for a user
% requested number, finds the LAST n spikes, and writes to a struct containing
% the data on all channels during those spike times. To correct for DC
% offset on any given spike, 10 samples to the left of the negative peak are
% averaged, and subtracted from the spike waveform.
% @Param analysisPath path to Vision output files.
%        example: '/Volumes/Analysis/2016-06-13-0/data000/data000'
% @Param dataPath path to raw .bin files.
%        example: '/Volumes/Data/2016-06-13-0/data000/'
% @Param writePath path of choosing in which to write the struct.
% @Param arrayID array ID of board used.
%        example: SB512-1 has '504'; SB512-3 has '1504' (char).
% @Param cellID cell number of interest from Vision.
%        example: 333 (int).
% @Param numSpikes number of spikes to write. As a default, this chooses the
%        LAST numSpikes, not the first.
% @Param leftSamples number of samples to save to the left of the spike peak.
% @Param traceLength total number of samples of the trace to save.
% @author Alex Gogliettino
% @date 2019-05-18.
% Constants.
% ---------
% SAMPLING_RATE: sampling rate of readout system. 20kHz.
% MEAN_SAMPLES: number of first samples to subtract out.
SAMPLING_RATE = 20000;
MEAN_LSAMPLES = 10; % Same as Vision.
% Add java path and master MATLAB branch.
javaaddpath('/Volumes/Lab/Development/Java/vision7/Vision.jar');
addpath(genpath('/Volumes/Lab/Users/AlexG/matlab-repos/master-matlab/'));
% Load data into datarun struct.
datarun = load_data(analysisPath);
datarun = load_neurons(datarun);
datarun = load_ei(datarun, 'all');
% For cell of interest, get the LAST numSpikes spikes of interest.
cellInd = find(datarun.cell_ids == cellID);
spikeTimes = flipud(datarun.spikes{cellInd});
% Convert spike times from seconds to samples.
spikeTimes = (spikeTimes * SAMPLING_RATE);
% For each of the spike times, write raw data to the struct.
rawFile = edu.ucsc.neurobiology.vision.io.RawDataFile(dataPath);
disp(rawFile);
% When querying the rawFile object, note two things:
%    1. Index with electrode+1 because of weird indexing bug and ttl.
%    2. Zero indexing for samples.
spikesStruct = {};
fprintf('Finding spikes for cellid %s...',num2str(cellID))
% Initialize raw data tensor, including ttl.
switch arrayID
    case '1504'
        numChannels = 520;
    case '504'
        numChannels = 513;
    otherwise
        error('Unknown arrayID passed.');
end
rawDataTensor = zeros(numSpikes,numChannels,traceLength);
for i = 1:numSpikes
    rawDataTensor(i,:,:) = ...
        rawFile.getData(spikeTimes(i) - leftSamples - 1, traceLength)';
end
% Compute the EI, and subtract off the mean for the individual spikes.
ei = squeeze(mean(rawDataTensor,1));
for i = 1:size(rawDataTensor,1)
    spikes = zeros(size(ei));
    for j = 1:size(rawDataTensor,2)
        spikes(j,:) = squeeze(rawDataTensor(i,j,:)) - mean(ei(j,1:MEAN_LSAMPLES));
    end
    spikesStruct{end+1} = spikes;
end
%{
[numSpikes,~,~] = size(rawDataTensor);
for i = 1:numSpikes
    spikes = zeros(numChannels,traceLength);
    for j = 1:numChannels
        % Correct for the amplifier offset.
        spikes(j,:) = squeeze(rawDataTensor(i,j,:)) - ...
                      mean(squeeze(rawDataTensor(i,j,1:MEAN_SAMPLES)));
    end
    spikesStruct{end+1} = spikes;
end
%}
fprintf('done.\n');
% Write out the spikes struct.
fprintf('Writing out the spikes struct...');
fnameout = [writePath 'spikesStruct_cellid' num2str(cellID)];
save(fnameout,'spikesStruct','-v7.3');
fprintf('done.\n');