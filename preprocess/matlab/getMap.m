% The ElectrodeMapFactory object handles the construction of ElectrodeMap objects
% from unique data set identifiers contained in the raw data header

elMapFactory = edu.ucsc.neurobiology.vision.electrodemap.ElectrodeMapFactory();

% Use the electrode map factory and array ID to get Vision to tell you what the electrode map was for the recording
elMap = elMapFactory.getElectrodeMap(header.getArrayID());
coordinates = elMap.toString;

elMap_dir = [util 'elMap.txt'];

x_offset = ones(nElec,2)*[16.25 8; 16.25 8];
posPhy = table2array(readtable(elMap_dir,'ReadVariableNames',false));
posLog = floor((posPhy./30.+x_offset)./2.+0.5);
posLog(:,[1 2]) = posLog(:,[2 1]);
posLin = single(sub2ind([16,32],posLog(:,1),posLog(:,2)));