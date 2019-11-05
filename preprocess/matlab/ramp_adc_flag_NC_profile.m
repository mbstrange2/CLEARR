%% re-convert data with ramp-ADC architecture
function [dataOut, bitTx] = ramp_adc_flag_NC_profile(dataIn, offsetPerChannel, posLin, B, FS, nwire)
% variable definitions and initialization
arrayIn = int16(zeros(16,32));
arrayOut = single(zeros(16,32));
arrayDecod = single(zeros(16,32));
arrayDecod21 = single(zeros(16,32));
arrayDecod22 = single(zeros(16,32));
arrayDecod41 = single(zeros(16,32));
arrayDecod42 = single(zeros(16,32));
arrayDecod43 = single(zeros(16,32));
arrayDecod44 = single(zeros(16,32));
mask21 = int16(zeros(16,32));
mask22 = int16(zeros(16,32));
mask41 = int16(zeros(16,32));
mask42 = int16(zeros(16,32));
mask43 = int16(zeros(16,32));
mask44 = int16(zeros(16,32));
nCodes = int16(max(max(abs(dataIn(2:end)))));
row = single(zeros(16,1));
col = single(zeros(1,32));
row21 = single(zeros(16,1));
col21 = single(zeros(1,32));
row22 = single(zeros(16,1));
col22 = single(zeros(1,32));
row41 = single(zeros(16,1));
col41 = single(zeros(1,32));
row42 = single(zeros(16,1));
col42 = single(zeros(1,32));
row43 = single(zeros(16,1));
col43 = single(zeros(1,32));
row44 = single(zeros(16,1));
col44 = single(zeros(1,32));
validOut = single(ones(16,32));
dataOut = int16(zeros(size(dataIn)));
i = logical(1);
ramp = single(zeros(2^B,1));
notConflict = logical(0);
notConflict21 = logical(0);
notConflict22 = logical(0);
notConflict41 = logical(0);
notConflict42 = logical(0);
notConflict43 = logical(0);
notConflict44 = logical(0);
bitTx = single(0);

%% test
% display('running adc_ramp in test mode')
% dataIn = rawData(41,:);

%% processing
arrayIn(posLin) = dataIn(2:end)-offsetPerChannel;
LSB = (2*FS/2^B);
minValue = single(min(min(arrayIn)));
maxValue = single(max(max(arrayIn)));
VRn = (floor(minValue/LSB) * LSB) - LSB;
VRp = (ceil(maxValue/LSB) * LSB) + LSB;
% linear ramp
ramp = (VRn:LSB:VRp);
%ramp = (-FS:2*FS/2^B:FS);
% half sine ramp
%ramp = FS*sin(-pi/2:pi/2^B:pi/2);

switch nwire
    
    case 1
        for i = 1:size(ramp,2)-1
            % find row and col vector per code per sample
            row = single(sum((arrayIn(:,:)>ramp(i))&(arrayIn(:,:)<=ramp(i+1)),2)>0);
            col = single(sum((arrayIn(:,:)>ramp(i))&(arrayIn(:,:)<=ramp(i+1)),1)>0);
            % save values only if #collisions less than nc
            notConflict = min(sum(row==1,1),sum(col==1,2)) == 1;
            % create a decoded array (cross row and column)
            % avoid overwriting strong decisions - no conflict in making them
            arrayDecod=row*col.*validOut*notConflict;
            % overwrite the output array with the decoded array
            arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i+1);
            % update the strong decision mask for each code looking at tmp
            validOut = validOut - arrayDecod * single(min(sum(row==1,1),sum(col==1,2)) == 1);
        end
        % counts number of flags and multiplies for the address word
        bitTx = sum(sum(validOut == 0)) * 9;
        
    case 2
        % activate only few channels per mask
         mask21(1:2:end) = 1;
         mask22(2:2:end-1) = 1;
        for i = 1:size(ramp,2)-1
            % find row and col vector per code per sample
            row21 = single(sum((arrayIn.*mask21>ramp(i))&(arrayIn.*mask21<=ramp(i+1)),2)>0);
            col21 = single(sum((arrayIn.*mask21>ramp(i))&(arrayIn.*mask21<=ramp(i+1)),1)>0);
            row22 = single(sum((arrayIn.*mask22>ramp(i))&(arrayIn.*mask22<=ramp(i+1)),2)>0);
            col22 = single(sum((arrayIn.*mask22>ramp(i))&(arrayIn.*mask22<=ramp(i+1)),1)>0);
            % save values only if #collisions less than nc
            notConflict21 = min(sum(row21==1,1),sum(col21==1,2)) == 1;
            notConflict22 = min(sum(row22==1,1),sum(col22==1,2)) == 1;
            % create a decoded array (cross row and column)
            % avoid overwriting strong decisions - no conflict in making them
            arrayDecod21 = row21*col21.*validOut*notConflict21;
            arrayDecod22 = row22*col22.*validOut*notConflict22;
            arrayDecod = arrayDecod21 + arrayDecod22;
            % overwrite the output array with the decoded array
            arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i+1);
            % update the strong decision mask for each code looking at tmp
            validOut = validOut - arrayDecod21 * single(min(sum(row21==1,1),sum(col21==1,2)) == 1)...
                - arrayDecod22 * single(min(sum(row22==1,1),sum(col22==1,2)) == 1);
        end
        % counts number of flags and multiplies for the address word
        bitTx = sum(sum(validOut == 0)) * 9;
        
    case 4
        % activate only few channels per mask
        mask41(1:4:end) = 1;
        mask42(2:4:end-1) = 1;
        mask43(3:4:end-2) = 1;
        mask44(4:4:end-3) = 1;
        for i = 1:size(ramp,2)-1
            % find row and col vector per code per sample
            row41 = single(sum((arrayIn.*mask41>ramp(i))&(arrayIn.*mask41<=ramp(i+1)),2)>0);
            col41 = single(sum((arrayIn.*mask41>ramp(i))&(arrayIn.*mask41<=ramp(i+1)),1)>0);
            row42 = single(sum((arrayIn.*mask42>ramp(i))&(arrayIn.*mask42<=ramp(i+1)),2)>0);
            col42 = single(sum((arrayIn.*mask42>ramp(i))&(arrayIn.*mask42<=ramp(i+1)),1)>0);
            row43 = single(sum((arrayIn.*mask43>ramp(i))&(arrayIn.*mask43<=ramp(i+1)),2)>0);
            col43 = single(sum((arrayIn.*mask43>ramp(i))&(arrayIn.*mask43<=ramp(i+1)),1)>0);
            row44 = single(sum((arrayIn.*mask44>ramp(i))&(arrayIn.*mask44<=ramp(i+1)),2)>0);
            col44 = single(sum((arrayIn.*mask44>ramp(i))&(arrayIn.*mask44<=ramp(i+1)),1)>0);
            % save values only if #collisions less than nc
            notConflict41 = min(sum(row41==1,1),sum(col41==1,2)) == 1;
            notConflict42 = min(sum(row42==1,1),sum(col42==1,2)) == 1;
            notConflict43 = min(sum(row43==1,1),sum(col43==1,2)) == 1;
            notConflict44 = min(sum(row44==1,1),sum(col44==1,2)) == 1;
            % create a decoded array (cross row and column)
            % avoid overwriting strong decisions - no conflict in making them
            arrayDecod41 = row41*col41.*validOut*notConflict41;
            arrayDecod42 = row42*col42.*validOut*notConflict42;
            arrayDecod43 = row43*col43.*validOut*notConflict43;
            arrayDecod44 = row44*col44.*validOut*notConflict44;
            arrayDecod = arrayDecod41 + arrayDecod42 + arrayDecod43 + arrayDecod44;
            % overwrite the output array with the decoded array
            arrayOut = (arrayOut.*abs(arrayDecod-1)) + arrayDecod*ramp(i+1);
            % update the strong decision mask for each code looking at tmp
            validOut = validOut - arrayDecod41 * single(min(sum(row41==1,1),sum(col41==1,2)) == 1)...
                - arrayDecod42 * single(min(sum(row42==1,1),sum(col42==1,2)) == 1)...
                - arrayDecod43 * single(min(sum(row43==1,1),sum(col43==1,2)) == 1)...
                - arrayDecod44 * single(min(sum(row44==1,1),sum(col44==1,2)) == 1);
        end
        % counts number of flags and multiplies for the address word
        bitTx = sum(sum(validOut == 0)) * 9;
        
    otherwise
        error('Number of wires not supported. Available options 1, 2, 4.')
end

% move from matrix of channels to vector of channels (use ElectrodeMap)
dataOut(2:end) = cast(arrayOut(posLin),'int16');
dataOut(1) = dataIn(1);
