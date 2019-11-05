function [hitRate, newIDidx] = compareNeurons(neuronFileRef, neuronFile,...
    nSpikestoTest, Tcorr, fs)

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
        if sum(newIDidx == j) == 0
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
            if(hitRate(i) > 0.5)
                newIDidx(i)   = j;
                break
            end
        end
        %fprintf(['j = ' num2str(j) '\n'])    
        end
    end
end
