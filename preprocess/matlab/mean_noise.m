%% determining average noise over channels using .noise files

analysisPath = '/Volumes/Scratch/Users/dantemur/analysis/2009-04-13-5/conflicts/data008/';
noisePath = [analysisPath 'original.noise'];
in = csvread(noisePath);
in = in(2:end);
noise_mean = mean(in)*4/2^12/270*1e6