%% re-convert data with ramp-ADC architecture
function [dataOut, nCollisions, activityFactor] = ramp_adc(dataIn, posLin, B, FS, nwire)
% variable definitions and initialization
arrayIn = single(zeros(16,32));
arrayOut = single(zeros(16,32));
arrayDecod = single(zeros(16,32));
nCodes = single(max(max(abs(dataIn(2:end)))));
row = single(zeros(16,1));
col = single(zeros(1,32));
validOut = single(ones(16,32));
dataOut = single(zeros(size(dataIn)));
i = single(1);
nCollisions = single(0);
activityFactor = single(0);
ramp = single(zeros(2*FS,1));

%% test
if(0)
    display('running adc_ramp in test mode')
    arrayIn = single(zeros(16,32));
    arrayIn(posLin) = rawData(25,2:end);
    ramp = (-FS:2*FS/2^B:FS)+0.5;
    i = round(size(ramp,2)/2);
end

%% processing
arrayIn(posLin) = dataIn(2:end);
% linear ramp
ramp = (-FS:2*FS/2^B:FS)+0.5;
% half sine ramp
%ramp = FS*sin(-pi/2:pi/2^B:pi/2);

% double wire
% for i = 1:size(ramp,2)-1
%     % find row and col vector per code per sample
%     row1 = single(sum((arrayIn(:,1:2:end)>ramp(i))&(arrayIn(:,1:2:end)<=ramp(i+1)),2)>0);
%     col1 = single(sum((arrayIn(1:2:end,:)>ramp(i))&(arrayIn(1:2:end,:)<=ramp(i+1)),1)>0);
%     row2 = single(sum((arrayIn(:,2:2:end)>ramp(i))&(arrayIn(:,2:2:end)<=ramp(i+1)),2)>0);
%     col2 = single(sum((arrayIn(2:2:end,:)>ramp(i))&(arrayIn(2:2:end,:)<=ramp(i+1)),1)>0);
%     % create a decoded array (cross row and column)
%     % avoid overwriting strong decisions - no conflict in making them
%     arrayDecod1 = row1*col1.*validOut;
%     arrayDecod2 = row2*col2.*validOut;
%     arrayDecod = arrayDecod1 + arrayDecod2;
%     % overwrite the output array with the decoded array
%     arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i);
%     % update the strong decision mask for each code looking at tmp
%     validOut = validOut - arrayDecod1 * single(min(sum(row1==1,1),sum(col1==1,2)) == 1)...
%         - arrayDecod2 * single(min(sum(row2==1,1),sum(col2==1,2)) == 1);
%     activityFactor = activityFactor + sum(row)+sum(col);
% end

% four wires
for i = 1:size(ramp,2)-1
    % find row and col vector per code per sample
    row1 = single(sum((arrayIn(:,1:4:end)>ramp(i))&(arrayIn(:,1:4:end)<=ramp(i+1)),2)>0);
    col1 = single(sum((arrayIn(1:4:end,:)>ramp(i))&(arrayIn(1:4:end,:)<=ramp(i+1)),1)>0);
    row2 = single(sum((arrayIn(:,2:4:end)>ramp(i))&(arrayIn(:,2:4:end)<=ramp(i+1)),2)>0);
    col2 = single(sum((arrayIn(2:4:end,:)>ramp(i))&(arrayIn(2:4:end,:)<=ramp(i+1)),1)>0);
    row3 = single(sum((arrayIn(:,3:4:end)>ramp(i))&(arrayIn(:,3:4:end)<=ramp(i+1)),2)>0);
    col3 = single(sum((arrayIn(3:4:end,:)>ramp(i))&(arrayIn(3:4:end,:)<=ramp(i+1)),1)>0);
    row4 = single(sum((arrayIn(:,4:4:end)>ramp(i))&(arrayIn(:,4:4:end)<=ramp(i+1)),2)>0);
    col4 = single(sum((arrayIn(4:4:end,:)>ramp(i))&(arrayIn(4:4:end,:)<=ramp(i+1)),1)>0);
    % create a decoded array (cross row and column)
    % avoid overwriting strong decisions - no conflict in making them
    arrayDecod1 = row1*col1.*validOut;
    arrayDecod2 = row2*col2.*validOut;
    arrayDecod3 = row3*col3.*validOut;
    arrayDecod4 = row4*col4.*validOut;
    arrayDecod = arrayDecod1 + arrayDecod2 + arrayDecod3 + arrayDecod4;
    % overwrite the output array with the decoded array
    arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i);
    % update the strong decision mask for each code looking at tmp
    validOut = validOut - arrayDecod1 * single(min(sum(row1==1,1),sum(col1==1,2)) == 1)...
        - arrayDecod2 * single(min(sum(row2==1,1),sum(col2==1,2)) == 1)...
        - arrayDecod3 * single(min(sum(row3==1,1),sum(col3==1,2)) == 1)...
        - arrayDecod4 * single(min(sum(row4==1,1),sum(col4==1,2)) == 1);
    activityFactor = activityFactor + sum(row)+sum(col);
end

% move from matrix of channels to vector of channels (use ElectrodeMap)
dataOut(2:end) = arrayOut(posLin);
dataOut(1) = dataIn(1);

% output number of collisions
nCollisions = sum(sum(validOut == 1));