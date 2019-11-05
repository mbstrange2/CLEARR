%% Set paths

% Path to Vision.jar
visionPath = '/home/dantemur/retina/Java/vision7/Vision.jar';
javaaddpath(visionPath);

% Path to Spectra
spectraPath = '/home/dantemur/retina/spikesorting/spectra';
cd spectraPath;
% Path to test
% rawDataDir = uigetdir('/home/dantemur/retina/', 'Select raw data directory'); 
testsPath = '/home/dantemur/retina/data/';
outputsPath = '/home/dantemur/retina/analysis/';
moviePath = '/home/dantemur/retina/utilities/RGB-8-4-0.48-11111.xml';

%% Run once
datasetPath = '2009-04-13-5/';
specsPath = '11b_20ks_12uV/data008';
rawDataPath = [testsPath datasetPath specsPath];
outDataPath = [outputsPath datasetPath specsPath];

mVision(rawDataPath, outDataPath, '', moviePath, 'all', 'all', 'DANTE_LOCAL')

%% Run once
datasetPath = '2009-04-13-5/';
specsPath = '10b_20ks_12uV/data008';
rawDataPath = [testsPath datasetPath specsPath];
outDataPath = [outputsPath datasetPath specsPath];

mVision(rawDataPath, outDataPath, '', moviePath, 'all', 'all', 'DANTE_LOCAL')

%% Run once
datasetPath = '2009-04-13-5/';
specsPath = '09b_20ks_12uV/data008';
rawDataPath = [testsPath datasetPath specsPath];
outDataPath = [outputsPath datasetPath specsPath];

mVision(rawDataPath, outDataPath, '', moviePath, 'all', 'all', 'DANTE_LOCAL')

%% Run once
datasetPath = '2009-04-13-5/';
specsPath = '08b_20ks_20uV/data008';
rawDataPath = [testsPath datasetPath specsPath];
outDataPath = [outputsPath datasetPath specsPath];

mVision(rawDataPath, outDataPath, '', moviePath, 'all', 'all', 'DANTE_LOCAL')

%% Run once
datasetPath = '2009-04-13-5/';
specsPath = '07b_20ks_40uV/data008';
rawDataPath = [testsPath datasetPath specsPath];
outDataPath = [outputsPath datasetPath specsPath];

mVision(rawDataPath, outDataPath, '', moviePath, 'all', 'all', 'DANTE_LOCAL')

%% Run once
datasetPath = '2009-04-13-5/';
specsPath = '06b_20ks_60uV/data008';
rawDataPath = [testsPath datasetPath specsPath];
outDataPath = [outputsPath datasetPath specsPath];

mVision(rawDataPath, outDataPath, '', moviePath, 'all', 'all', 'DANTE_LOCAL')