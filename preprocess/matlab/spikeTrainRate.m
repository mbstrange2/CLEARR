function spikeTrainRate

setPaths;

datarunOrig=load_data('2015-11-09-3/data000');
datarunOrig=load_params(datarunOrig);
datarunOrig=load_neurons(datarunOrig);

datarunMod=load_data('/Volumes/Scratch/Users/dantemur/analysis/2015-11-09-3/conflicts/data000/data000');
datarunMod=load_params(datarunMod);
datarunMod=load_neurons(datarunMod);


for i = 1
    ST1 = datarunOrig.spikes{i};
    for k = 1
        ST2 = datarunMod.spikes{k};
        
    end
end

[probabilities, bins, norm] = autocorrelation(datarunOrig.spikes{1}, 1, 100, 1.8e3);