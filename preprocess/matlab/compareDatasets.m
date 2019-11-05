%% Author: Dante Muratore - dantemur@stanford.edu
% 20-Apr-2018
%
% This function compares previously sorted White Noise Analysis Datasets to
% find corresponding cells. 
clear all;clc;close all;

analysisRepository = '/Volumes/Scratch/Users/dantemur/analysis/';
dataset = 'data000';
analysisPathRef = [analysisRepository '2015-11-09-3/short_orig/' dataset];
analysisPath = [analysisRepository '2015-11-09-3/short_flag_NC/' dataset];
util = '/Volumes/Lab/Users/dantemur/readout/matlab/utilities/';
setPaths(analysisPathRef,analysisPath,util)


neuronPathRef = [analysisPathRef '/' dataset '.neurons'];
neuronFileRef = edu.ucsc.neurobiology.vision.io.NeuronFile(neuronPathRef);
neuronPath = [analysisPath '/' dataset '.neurons'];
neuronFile = edu.ucsc.neurobiology.vision.io.NeuronFile(neuronPath);

% Methods of interest
% 
% neuronFile.getHeader(): returns the file header.
% neuronFile.getTTLTimes(): returns the TTL times in samples.
% neuronFile.getIDList(): returns a list of all the neuron IDs in this neuron file
% neuronFile.getSpikeTimes(neuronId): returns the list of spike times for the neuron whose ID is neuronID. Spike times are returned in samples.
% neuronFile.getNeuronIDElectrode(neuronId): returns the electrode on which the spikes for the neuron with ID neuronId were seen.

Tcorr = 2.5e-3;
fs = 20e3;
IDsRef  = neuronFileRef.getIDList();
IDs     = neuronFile.getIDList();
nNeuronsRef = size(IDsRef,1);
nNeurons    = size(IDs,1);
C = zeros(nNeuronsRef,2*Tcorr*fs+1);
AC = zeros(nNeuronsRef,2*Tcorr*fs+1);
lagsC = zeros(nNeuronsRef,2*Tcorr*fs+1);
lagsAC = zeros(nNeuronsRef,2*Tcorr*fs+1);
hitRate = zeros(nNeuronsRef,1);
newIDidx  = NaN(nNeuronsRef,1);
strongIDidx  = NaN(nNeuronsRef,1);
nSpikestoTest = 100;
cellsRight = 0;

for i = 1:nNeuronsRef
    fprintf(['i = ' num2str(i) '\n'])
    % generate spike train for neuron i in the reference
    neuronRefTime  = neuronFileRef.getSpikeTimes(IDsRef(i));
    spikeTrainRef = zeros(neuronRefTime(nSpikestoTest),1);
    spikeTrainRef(neuronRefTime(1:nSpikestoTest),1) = 1;
    % calculate autocorrelation function and get total spikes in +-0.5ms
    [AC,lagsAC]  = xcorr(spikeTrainRef,Tcorr*fs);
    totSpikesRef = sum(AC(round(end/2)-10:1:round(end/2)+10));
    for j = 1:nNeurons
        if sum(strongIDidx == j) == 0
        %generate spike train for neuron j in the comparison
        neuronTime = neuronFile.getSpikeTimes(IDs(j));
        spikeTrain = zeros(neuronTime(nSpikestoTest),1);
        spikeTrain(neuronTime(1:nSpikestoTest),1) = 1;
        % calculate correlation to reference and get hitRate, comparing
        % total spikes in +-0.5ms
        [Ctmp,lagsCtmp] = xcorr(spikeTrainRef,spikeTrain,Tcorr*fs);
        hitRatetmp      = sum(Ctmp(round(end/2)-10:1:round(end/2)+10))/totSpikesRef;
        if hitRatetmp > hitRate(i)
            hitRate(i)  = hitRatetmp;
            C(i,:)      = Ctmp;
            lagsC(i,:)  = lagsCtmp;
            newIDidx(i)   = j;
            if(hitRate(i) > 0.35)
                strongIDidx(i)   = j;
                cellsRight = cellsRight + 1;
                break
            end
        end
        %fprintf(['j = ' num2str(j) '\n'])    
        end
    end
end

hitCells = cellsRight/nNeuronsRef

bins = 0:0.1:1;
hist(hitRate,bins)
grid on
